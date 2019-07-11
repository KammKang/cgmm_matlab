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
M=4;
threshold=1.3;
% You just neet to give your wav and M and repalace them here.
%% enframe and do fft
frame_length = 400;
frame_shift = 160;
fft_len = 1024;%2 倍的帧长
[frames, ffts] = multi_fft(wav, frame_length, frame_shift, fft_len);
[M,T,F] = size(ffts);
%ffts
output = zeros(T, F);
%% Estimate the TF-SPP and spacial covariance matrix for noisy speech and noise 
for t = 1:25:T-25 

    [lambda_v, lambda_y, Ry, Rv] = est_cgmm(ffts(:,t:t+25,:));
    ffts_ffts = outProdND(ffts(:,t:t+25,:));
    outer=outProdND(ffts(:,t:t+25,:)); %M*T*F-->M*M*T*F
    R_y = squeeze(mean(outer, 3));
    Rx = Ry -Rv;         %trade off. Rx may be not positive definite 正定

    [M, T, F]  = size(ffts(:,t:t+25,:)); %fft bins number
    d = zeros(M, F);         %steering vectors
                      %mvdr beamforming weight 
    %% Get steering vectors d using eigvalue composition 
    for f= 1:F
        [V, D] = eig(squeeze(Rx(:, :, f)));
        max = D(1,1)
        index =1
        for i = 1:(size(D,1))
            if(D(i,i)>max)
                max = D(i,i)
                index = i
            end
        end
        d(:,f) = D(:,index);
    end
    %% Do MVDR beamforming
    output(t:t+25,:) = mvdr(ffts(:,t:t+25,:), Ry, d);
end
%% Reconstruct time domain signal using overlap and add;
output = [output, fliplr(conj(output(:, 2:end-1)))];
rec_frames = real(ifft(output, fft_len, 2));
rec_frames = rec_frames(:,1:frame_length);
sig = overlapadd(rec_frames, hamming(frame_length, 'periodic'), frame_shift);

audiowrite('output_Suv1.wav', sig, 16000);
