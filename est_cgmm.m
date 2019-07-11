function [ A_y, A_v, R_xn, R_n ] = est_cgmm( ffts )
%EST_CGMM is used to estimate the Complex GMM parameters 
%and generate the mask for nosie only and noisy t-f bins
%   ffts: M*L*(fft_len/2+1), the multi-channel fft matrix
%   lambda_v: the mask for noise only t-f bins
%   lambda_y: the mask for noisy t-f bins
%   Ry, Rv: the spacial covariance matrix of noisy and noise;;
%           M*M*F;

[M, T, F ] = size(ffts);

lambda_v = zeros(T, F);
lambda_y =zeros(T, F);
outer=outProdND(ffts); %M*T*F-->M*M*T*F

Ry = squeeze(mean(outer, 3));%去除 dim=1 的维度
R_n = zeros([M, M, F]);
Rv = eye(M);
Rv = reshape(Rv, [size(Rv, 1), size(Rv, 2), 1]);
Rv = repmat(Rv, [1, 1, F]);
phi_y = ones(T, F);
phi_v = ones(T, F);

file =fopen('testVector.txt','w');

for iter=1:10
    for f=1:F
        Ry_f = Ry(:, :, f);
        Rv_f = Rv(:, :, f);
        if rcond(Ry_f) < 0.0001
            Ry_f = Ry_f + rand(M)*0.0001;
        end
        if rcond(Rv_f) < 0.0001
            Rv_f = Rv_f + rand(M)*0.0001;
        end
        invRy_f = inv(Ry_f);
        invRv_f = inv(Rv_f);
        y_tf = ffts(:,:, f);% M*T*1
        
        y_y_tf = outProdND(y_tf);% M*M*T
        
      
        sum_y = zeros(M);
        sum_v = zeros(M);
        acc_n = zeros(M);%
        e= eye(M)*0.00000;
        for t = 1:T
            phi_y(t, f) = (1/M)*(trace(y_y_tf(:, :, t)*invRy_f));
            phi_v(t, f) = (1/M)*(trace(y_y_tf(:, :, t)*invRv_f)); %eq11   
            %计算概率
            kernel_y = y_tf(:, t)' * (1/phi_y(t, f))*invRy_f * y_tf(:, t);
            kernel_v = y_tf(:, t)' * (1/phi_v(t, f))*invRv_f * y_tf(:, t);
            
            p_y(t, f) = exp(-kernel_y)/(pi*det(phi_y(t, f)*Ry_f));
            p_v(t, f) = exp(-kernel_v)/(pi*det(phi_v(t, f)*Rv_f));%高斯一波
            %非统计版
            lambda_y(t, f) = p_y(t, f) / (p_y(t, f)+p_v(t, f)); 
            lambda_v(t, f) = p_v(t, f) / (p_y(t, f)+p_v(t, f));%eq10
            sum_y = sum_y + lambda_y(t, f)/phi_y(t, f)*y_y_tf(:, :, t);%for eq12
            sum_v = sum_v + lambda_v(t, f)/phi_v(t, f)*y_y_tf(:, :, t);
            
            acc_n = acc_n + lambda_v(t, f)*y_y_tf(:, :, t); %for eq(4)
            
        end
        R_n(:, :, f) = 1/sum(lambda_y(:, f)) * acc_n; %eq(4)
        
        tmp_Ry_f = 1/sum(lambda_y(:, f)) * sum_y;%eq 12
        tmp_Rv_f = 1/sum(lambda_v(:, f)) * sum_v;
        
       % [V1, D1] = eig(squeeze(tmp_Ry_f));
       % [V2, D2] = eig(squeeze(tmp_Rv_f));
        %
        %fprintf(file,'%e %e %e %e\r\n',V1(1,1),V1(1,2),V1(2,1),V1(2,2));
       % entropy1 = -diag(D1, 0)'/sum(diag(D1, 0)) * log(diag(D1, 0)/sum(diag(D1, 0)));
       % entropy2 = -diag(D2, 0)'/sum(diag(D2, 0)) * log(diag(D2, 0)/sum(diag(D2, 0)));
        %if entropy1 > entropy2
        %    Ry(:, :, f) = tmp_Rv_f;
        %    Rv(:, :, f) = tmp_Ry_f;
       % else
            Ry(:, :, f) = tmp_Ry_f;
            Rv(:, :, f) = tmp_Rv_f;
        %end
       A_y(f) = sum(lambda_y(:, f));
       A_v(f) = sum(lambda_v(:, f));
   end
    
    %Q = sum(sum(lambda_y .* log(p_y+0.001) + lambda_v .* log(p_v+0.001)));%eq9
    figure(1)
    imagesc(real([flipud(lambda_y');flipud(lambda_v')]));
end
for f=1:F
    eig_value1 = eig(Ry(:, :, f));
    eig_value2 = eig(Rv(:, :, f));
    en_noise = -eig_value1' / sum(eig_value1) * log(eig_value1 / sum(eig_value1));
    en_noisy = -eig_value2' / sum(eig_value2) * log(eig_value2 / sum(eig_value2));
    if en_noise > en_noisy
        Rn = Ry(:, :, f);
        Ry(:, :, f) = Rv(:, :, f);
        Rv(:, :, f) = Rn;
    end
end
R_xn = Ry;

end

