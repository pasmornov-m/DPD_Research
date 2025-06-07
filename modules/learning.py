import torch
from modules.utils import to_torch_tensor, check_early_stopping
from modules.metrics import compute_mse, add_complex_noise

def optimize_dla_grad(input_data, target_data, dpd_model, pa_model, epochs=100000, learning_rate=1e-3, 
                      add_noise=False, snr=None, fs=None, bw=None):
    input_data, target_data = map(to_torch_tensor, (input_data, target_data))

    optimizer = torch.optim.Adam(dpd_model.parameters(), lr=learning_rate, amsgrad=True)
    for param in pa_model.parameters():
        param.requires_grad = False
            
    for epoch in range(epochs):
        optimizer.zero_grad()

        dpd_output = dpd_model.forward(input_data)
        pa_output = pa_model.forward(dpd_output)

        if add_noise:
            if snr is None or fs is None or bw is None:
                raise ValueError("SNR, fs, and bw must be provided when add_noise=True")
            pa_output = add_complex_noise(pa_output, snr, fs, bw)

        loss = compute_mse(pa_output, target_data)
        loss.backward()
        optimizer.step()

        if epoch%99==0:
                print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")

    print("DLA optimization completed.")


def optimize_ila_grad(dpd_model, input_data, output_data, gain, epochs=100000, learning_rate=0.01, pa_model=None, 
                      add_noise=False, snr=None, fs=None, bw=None):   
    input_data = to_torch_tensor(input_data)
    
    if pa_model:
        print("Compute on pa_model")
        output_data = (pa_model.forward(input_data) / gain).detach()
    else:
        print("Compute on presaved output_data")
        output_data = to_torch_tensor(output_data) / gain    

    optimizer = torch.optim.Adam(dpd_model.parameters(), lr=learning_rate, amsgrad=True)
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = dpd_model.forward(output_data)
        if add_noise:
            if snr is None or fs is None or bw is None:
                raise ValueError("SNR, fs, and bw must be provided when add_noise=True")
            output = add_complex_noise(output, snr, fs, bw)
        loss = compute_mse(output, input_data)
        loss.backward()
        optimizer.step()

        if epoch%99==0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")

    print("ILA-DPD Training Complete.")


def ilc_signal_grad(input_data, target_data, pa_model, epochs=1000000, learning_rate=0.1, 
                    add_noise=False, snr=None, fs=None, bw=None):
    input_data, target_data = map(to_torch_tensor, (input_data, target_data))
    u = torch.nn.Parameter(input_data.clone(), requires_grad=True)

    optimizer = torch.optim.AdamW([u], lr=learning_rate, amsgrad=True)
    for param in pa_model.parameters():
        param.requires_grad = False

    for epoch in range(epochs):
        optimizer.zero_grad()
        pa_output = pa_model.forward(u)

        if add_noise:
            if snr is None or fs is None or bw is None:
                raise ValueError("SNR, fs, and bw must be provided when add_noise=True")
            pa_output = add_complex_noise(pa_output, snr, fs, bw)

        loss = compute_mse(pa_output, target_data)
        loss.backward()
        optimizer.step()

        if epoch%99==0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")

    return u.detach()

