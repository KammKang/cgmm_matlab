function [ Ay,Av,Ry_r,Rv_r,R_n_t ] = est_cgmm_T(ffts,Ay,Av,Ry_r,Rv_r,R_n_t,frame_count)
%EST_CGMM is used to estimate the Complex GMM parameters 
%and generate the mask for nosie only and noisy t-f bins
%   ffts: M*L*(fft_len/2+1), the multi-channel fft matrix
%   lambda_v: the mask for noise only t-f bins
%   lambda_y: the mask for noisy t-f bins
%   Ry, Rv: the spacial covariance matrix of noisy and noise;;
%           M*M*F;
%   Ry_r = Ry_r + Ry_f    : 之前帧+当前帧
%   Rv_r = Rv_r + Rv_f
%
%
[M, T, F ] = size(ffts);

lambda_v = zeros(T, F);
lambda_y =zeros(T, F);
%M*T*F-->M*M*T*F

phi_y = ones(T, F);
phi_v = ones(T, F);
% bug 每轮iter 相同
for iter=1:4
    %disp(T)
    for f=1:F  
        Ry_f = Ry_r(:, :, f);
        Rv_f = Rv_r(:, :, f);
        % rcond(Ry_f) = |Ry_f| *|Ry_f-1|
        if rcond(Ry_f) < 0.0001
            Ry_f = Ry_f + rand(M)*0.0001;
        end
        if rcond(Rv_f) < 0.0001
            Rv_f = Rv_f + rand(M)*0.0001;
        end
        invRy_f = inv(Ry_f);
        invRv_f = inv(Rv_f);
        y_tf = ffts(:,1:T,f);% M*T*1
        
        y_y_tf = outProdND(y_tf);% M*M*T
       
       
        sum_y = zeros(M);
        sum_v = zeros(M);
        acc_n = zeros(M);%
        e= eye(M)*0.00000;
        for t = 1:T  % 当前帧，todo: 当前帧段（一下过来很多帧）。
            phi_y(t, f) = (1/M)*(trace(y_y_tf(:, :,t)*invRy_f));
            phi_v(t, f) = (1/M)*(trace(y_y_tf(:, :, t)*invRv_f)); %eq11   
            %计算概率
            kernel_y = y_tf(:, t)' * (1/phi_y(t, f))*invRy_f * y_tf(:, t);%
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
        % 加步进学习。 R_n_t = R_n_t * (t)/T + （1）/T（R_n_new）
        if frame_count>=100 
            frame_count =100;
        end
        %有问题。
        R_n_t(:, :, f) = 1/(frame_count)*(frame_count-1)*R_n_t(:, :, f)  +  1/(frame_count)*1/sum(lambda_y(:, f)) * acc_n; %eq(4)
        % lambda_y :只有T个数据有值，*sum_
        tmp_Ry_f = Ay(f)/(Ay(f)+sum(lambda_y(:, f)))*Ry_r(:, :, f) + 1/(Ay(f)+sum(lambda_y(:, f)))* sum_y;%eq 12
        tmp_Rv_f = Av(f)/(Av(f)+sum(lambda_v(:, f)))*Rv_r(:, :, f)+  1/(Av(f)+sum(lambda_v(:, f))) * sum_v;
        %待删减
        if rcond(tmp_Ry_f)<0.0001 
            tmp_Ry_f = tmp_Ry_f+rand(M)*0.00001;
        end
        if det(tmp_Rv_f)==0
            tmp_Rv_f = tmp_Rv_f+rand(M)*0.00001;
        end
        %tmp_Ry_f 出现问题。奇异值
        [V1, D1] = eig(squeeze(tmp_Ry_f));
        [V2, D2] = eig(squeeze(tmp_Rv_f));
        %
        entropy1 = -diag(D1, 0)'/sum(diag(D1, 0)) * log(diag(D1, 0)/sum(diag(D1, 0)));
        entropy2 = -diag(D2, 0)'/sum(diag(D2, 0)) * log(diag(D2, 0)/sum(diag(D2, 0)));
        if entropy1 > entropy2
            Ry_r(:, :, f) = tmp_Rv_f;
            Rv_r(:, :, f) = tmp_Ry_f;
        else
            Ry_r(:, :, f) = tmp_Ry_f;
            Rv_r(:, :, f) = tmp_Rv_f;
        end
    end
    
    %Q = sum(sum(lambda_y .* log(p_y+0.001) + lambda_v .* log(p_v+0.001)))%eq9
    %figure(1)
    %imagesc(real([flipud(lambda_y');flipud(lambda_v')]));
end
for f=1:F
    Ay(f) = Ay(f)+sum(lambda_y(:,f));
    Av(f) = Av(f)+sum(lambda_v(:,f));
end
end

