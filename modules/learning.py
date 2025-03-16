import torch
from modules.utils import to_torch_tensor, check_early_stopping
from modules.metrics import compute_mse, add_complex_noise

def optimize_dla_grad(input_data, target_data, dpd_model, pa_model, epochs=100000, learning_rate=1e-3, 
                      add_noise=False, snr=None, fs=None, bw=None):
    input_data, target_data = map(to_torch_tensor, (input_data, target_data))

    optimizer = torch.optim.AdamW(dpd_model.parameters(), lr=learning_rate, eps = 1e-16, betas=(0.9, 0.999), weight_decay=1e-5, amsgrad=True)
    for param in pa_model.parameters():
        param.requires_grad = False
            
    for epoch in range(epochs):
        optimizer.zero_grad()
        dpd_output = dpd_model.compute_output(input_data)
        pa_output = pa_model.compute_output(dpd_output)

        if add_noise:
            if snr is None or fs is None or bw is None:
                raise ValueError("SNR, fs, and bw must be provided when add_noise=True")
            pa_output = add_complex_noise(pa_output, snr, fs, bw)

        loss = compute_mse(pa_output, target_data)
        loss.backward()
        optimizer.step()
        current_loss = loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {current_loss}")

    print("DLA optimization completed.")


def optimize_ila_grad(dpd_model, input_data, output_data, gain, epochs=100000, learning_rate=0.01, pa_model=None, 
                      add_noise=False, snr=None, fs=None, bw=None):   
    input_data = to_torch_tensor(input_data)
    
    if pa_model:
        print("Compute on pa_model")
        output_data = (pa_model.compute_output(input_data) / gain).detach()
    else:
        print("Compute on presaved output_data")
        output_data = to_torch_tensor(output_data) / gain

    if add_noise:
        if snr is None or fs is None or bw is None:
            raise ValueError("SNR, fs, and bw must be provided when add_noise=True")
        output_data = add_complex_noise(output_data, snr, fs, bw)

    dpd_model.optimize_coefficients_grad(output_data, input_data, epochs, learning_rate)
    
    print("ILA-DPD Training Complete.")


def ilc_signal_grad(input_data, target_data, pa_model, max_iterations=1000000, learning_rate=0.1, 
                    add_noise=False, snr=None, fs=None, bw=None):
    input_data, target_data = map(to_torch_tensor, (input_data, target_data))
    u = torch.nn.Parameter(input_data.clone(), requires_grad=True)

    optimizer = torch.optim.AdamW([u], lr=learning_rate, amsgrad=True)
    for param in pa_model.parameters():
        param.requires_grad = False

    for iteration in range(max_iterations):
        optimizer.zero_grad()
        pa_output = pa_model.compute_output(u)

        if add_noise:
            if snr is None or fs is None or bw is None:
                raise ValueError("SNR, fs, and bw must be provided when add_noise=True")
            pa_output = add_complex_noise(pa_output, snr, fs, bw)

        loss = compute_mse(pa_output, target_data)
        loss.backward()
        optimizer.step()

        print(f"Iteration {iteration + 1}/{max_iterations}, Loss: {loss}")

    return u.detach()

