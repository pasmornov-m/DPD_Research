import torch
from torch import nn
from modules.utils import to_torch_tensor
from modules.metrics import compute_mse, add_complex_noise

def optimize_dla(input_data, target_data, dpd_model, pa_model, epochs=100000, learning_rate=1e-3, 
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

        if epoch%100==0:
                print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")

    print("DLA optimization completed.")


def optimize_ila(dpd_model, input_data, output_data, gain, epochs=100000, learning_rate=0.01, pa_model=None, 
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

        if epoch%100==0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")

    print("ILA-DPD Training Complete.")


def ilc_signal(input_data, target_data, pa_model, epochs=1000000, learning_rate=0.1, 
                    add_noise=False, snr=None, fs=None, bw=None):
    input_data, target_data = map(to_torch_tensor, (input_data, target_data))
    u = torch.nn.Parameter(input_data.clone(), requires_grad=True)
    optimizer = torch.optim.Adam([u], lr=learning_rate)
    with torch.no_grad():
        for param in pa_model.parameters():
            param.requires_grad = False
        
    if isinstance(pa_model, nn.Module):
        pa_model = pa_model.eval()

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

        if epoch%100==0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")

    return u.detach()


def net_train(net,
              dataloader,
              optimizer,
              criterion,
              grad_clip_val):
    net = net.train()
    losses = []
    for features, targets in dataloader:
        optimizer.zero_grad()
        out = net(features)
        loss = criterion(out, targets)
        loss.backward()
        if grad_clip_val != 0:
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip_val)
        optimizer.step()
        loss.detach()
        losses.append(loss.item())
    loss = sum(losses) / len(losses)
    return net, loss


def net_eval(net, dataloader, criterion, metric_criterion=None):
    net = net.eval()
    with torch.no_grad():
        losses = []
        metric_losses = []
        for features, targets in dataloader:
            outputs = net(features)
            loss = criterion(outputs, targets)
            metric_loss = metric_criterion(outputs, targets)
            losses.append(loss.item())
            metric_losses.append(metric_loss)
    avg_loss = sum(losses) / len(losses)
    avg_metric_loss = sum(metric_losses) / len(metric_losses)
    return avg_loss, avg_metric_loss


def train(net, 
          criterion, 
          optimizer,
          train_loader, 
          val_loader, 
          n_epochs,
          metric_criterion,
          grad_clip_val=0):
    print("===Start training===")
    for epoch in range(n_epochs):
        net, train_loss = net_train(net=net,
                        optimizer=optimizer,
                        criterion=criterion,
                        dataloader=train_loader,
                        grad_clip_val=grad_clip_val)

        val_loss, val_metric_loss = net_eval(net=net,
                                             criterion=criterion,
                                             dataloader=val_loader,
                                             metric_criterion=metric_criterion)

        print(f"Epoch {epoch:02d} — train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, val_NMSE: {val_metric_loss:.2f}")

    print("===Training complete===")


def net_inference(net, x):
    net = net.eval()
    with torch.no_grad():
        y = net(x)
    return y
