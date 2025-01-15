clear all
close all

% Data loading
load('y_in_tensor.mat');
y_in_tensor = y_in_tensor.';
load("y_out_tensor.mat");
y_out_tensor = y_out_tensor.';
load("y_target.mat");
y_target = y_target.';
load("u_k_pa.mat");
u_k_pa = u_k_pa.';

load("y_linearized_dla_grad.mat");
y_linearized_dla_grad = y_linearized_dla_grad.';
load("y_linearized_ila_grad.mat");
y_linearized_ila_grad = y_linearized_ila_grad.';
load("y_linearized_ilc_grad.mat");
y_linearized_ilc_grad = y_linearized_ilc_grad.';


% ACPR calculating
fs = 800e6;
bw_main_ch = 200e6;
bw_sub_ch = 20e6;
n_sub_ch = 10;
N_fft = 2560;

hACPR = comm.ACPR('SampleRate', fs, ...
                  'PowerUnits', 'dBW', ...
                  'MainChannelFrequency', 0, ...
                  'MainMeasurementBandwidth', bw_main_ch, ...
                  'AdjacentChannelOffset', [-bw_main_ch, bw_main_ch], ...
                  'AdjacentMeasurementBandwidth', bw_main_ch, ...
                  'MainChannelPowerOutputPort', true, ...
                  'AdjacentChannelPowerOutputPort', true, ...
                  'SpectralEstimation', 'Specify window parameters', ...
                  'SegmentLength', N_fft, ...
                  'OverlapPercentage', 60, ...
                  'Window', 'Blackman-Harris', ...
                  'FFTLength', 'Custom', ...
                  'CustomFFTLength', N_fft);

[ACPR_in, mainChannelPower_in, adjChannelPower_in] = hACPR(y_in_tensor);
[ACPR_out, mainChannelPower_out, adjChannelPower_out] = hACPR(y_out_tensor);
[ACPR_target, mainChannelPower_target, adjChannelPower_target] = hACPR(y_target);
release(hACPR);
[ACPR_u_k_pa, mainChannelPower_u_k_pa, adjChannelPower_u_k_pa] = hACPR(u_k_pa);
release(hACPR);

[ACPR_dla_grad, mainChannelPower_dla_grad, adjChannelPower_dla_grad] = hACPR(y_linearized_dla_grad);
[ACPR_ila_grad, mainChannelPower_ila_grad, adjChannelPower_ila_grad] = hACPR(y_linearized_ila_grad);
[ACPR_ilc_grad, mainChannelPower_ilc_grad, adjChannelPower_ilc_grad] = hACPR(y_linearized_ila_grad);


fprintf('ACPR in:           %.2f %.2f\n', ACPR_in(1), ACPR_in(2))
fprintf('ACPR out:          %.2f %.2f\n', ACPR_out(1), ACPR_out(2))
% fprintf('ACPR target: %.2f %.2f\n', ACPR_target(1), ACPR_target(2))
fprintf('ACPR ilc signal:   %.2f %.2f\n', ACPR_u_k_pa(1), ACPR_u_k_pa(2))
fprintf('ACPR dla grad:     %.2f %.2f\n', ACPR_dla_grad(1),ACPR_dla_grad(2))
fprintf('ACPR ila grad:     %.2f %.2f\n', ACPR_ila_grad(1), ACPR_ila_grad(2))
fprintf('ACPR ilc grad:     %.2f %.2f\n', ACPR_ilc_grad(1), ACPR_ilc_grad(2))
