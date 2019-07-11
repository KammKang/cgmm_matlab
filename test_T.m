%% Test scripts 
%% Author Sining Sun (NWPU)
% snsun@nwpu-aslp.org
clc
clear

%% Load the test multi-channel test data
I = 2; %channels number
for i = 1:4
    wav_all(:, i) = audioread(['test_wav/test3/16k_ch' int2str(i) '.wav']);
end
wav= wav_all; % we do not use ch2 because of bad quality
M=4
% You just neet to give your wav and M and repalace them here.
%% enframe and do fft
frame_length = 400;
frame_shift = 160;
fft_len = 512;
[frames, ffts] = multi_fft(wav, frame_length, frame_shift, fft_len);
[M,T,F] = size(ffts);
R_n_t = zeros([M,M,F]);
outer = outProdND(ffts);
Ry_r = ones([M,M,F]);%去除 dim=1 的维度
Rn = zeros([M, M, F]);
Rv_r = eye(M);
Rv_r = reshape(Rv_r, [size(Rv_r, 1), size(Rv_r, 2), 1]);
Rv_r = repmat(Rv_r, [1, 1, F]);
A = zeros(1,F);
frame_count = 1 ;
%% Estimate the TF-SPP and spacial covariance matrix for noisy speech and noise 
[M, T, F]  = size(ffts); %fft bins number
d = zeros(M, F);         %steering vectors
w = d;                   %mvdr beamforming weight 
output = zeros(T, F);    %beamforming outputs
real_output = zeros(T,fft_len);
Ay=zeros(F,1);
Av=zeros(F,1);
[Ay,Av,Ry_r, Rv_r] = est_cgmm(ffts(:,1:100,:));
Rx = Ry_r - Rv_r;
for f= 1:F
        [V, D] = eig(squeeze(Rx(:, :, f)));
        d(:, f) = V(:, 1);
end
output(1:100,:) = mvdr(ffts(:,1:100,:), Ry_r, d);
for t= 101:25:T-25 
        a = t;
        b = a+24;
        [Ay,Av,Ry_r, Rv_r,Rn] = est_cgmm_T_new(ffts(:,a:b,:),Ay,Av,Ry_r,Rv_r);    
        frame_count=frame_count+1;
        %trade off. Rx may be not positive definite 正定
        Rx = Ry_r -Rn;
%% Get steering vectors d using eigvalue composition 
        for f= 1:F
        [V, D] = eig(squeeze(Rx(:, :, f)));
        A=[];
        size(D(:,1))
        for i = 1:size(D(:,1))

            A(i) = D(i,i);
        end
        A = sort(A,'descend');
        N=[];
        for i =1:size(A,2)-1
            if(A(i)/A(i+1)>1.3)
                N(i)=A(i);
            end
        end
        Rnew = zeros(M,M);
        for i=1:size(N,2)
            for j =1:size(V,2)
                if N(i)==D(j,j)
                    Rnew(:,i) = V(:,j);
                end
            end
        end
        [U,S,V] = svd(Rnew);
        S_index=1;
        S_max=S(1);
        for s=1:size(S)
            if(S(s)>S_max)
                S_index = s;
                S_max = S(s);
            end
        end
        d(:,f) = U(:,S_index);
    end
%% Do MVDR beamforming
        R_n_t = mean(outProdND(ffts(:,a:b,:)),3);
       
        output(a:b,:) = mvdr(ffts(:,a:b,:), Ry_r, d);
        a

%% Reconstruct time domain signal using overlap and add;
   
end
output = [output, fliplr(conj(output(:, 2:end-1)))];

rec_frames = real(ifft(output, fft_len, 2));
rec_frames = rec_frames(:,1:frame_length);
sig = overlapadd(rec_frames, hamming(frame_length, 'periodic'), frame_shift);

audiowrite('output_4mic7.wav', sig, 16000);
