import torch
from torch import nn
from modules.metrics import compute_mse
from modules.utils import timer_decorator, complex_handler
# import time
# from datetime import timedelta


@timer_decorator
def ilc_signal(input_data, target_data, pa_model, epochs=100, learning_rate=0.1):
    u = torch.nn.Parameter(input_data.clone(), requires_grad=True)
    optimizer = torch.optim.Adam([u], lr=learning_rate)

    pa_model = pa_model.eval()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pa_output = pa_model.forward(u)
        loss = compute_mse(pa_output, target_data)
        loss.backward()
        optimizer.step()

        if epoch%100==0 or epoch == epochs - 1:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")
    return u.detach()


def net_train(net,
              dataloader,
              optimizer,
              criterion,
              grad_clip_val):
    net = net.train()
    losses = 0
    for features, targets in dataloader:
        optimizer.zero_grad()
        out = net(features)
        loss = criterion(out, targets)
        loss.backward()
        if grad_clip_val != 0:
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip_val)
        optimizer.step()
        losses += loss.item() * features.size(0)
    losses /= len(dataloader.dataset)
    return net, loss


def net_eval(net, dataloader, scheduler, criterion, metric_criterion=None):
    net = net.eval()
    with torch.no_grad():
        val_loss = 0
        metric_val_loss = 0
        for features, targets in dataloader:
            outputs = net(features)
            loss = criterion(outputs, targets)
            metric_loss = metric_criterion(outputs, targets)
            val_loss += loss.item() * features.size(0)
            metric_val_loss += metric_loss.item() * features.size(0)
    avg_loss = val_loss / len(dataloader.dataset)
    avg_metric_loss = metric_val_loss / len(dataloader.dataset)
    
    if scheduler:
        scheduler.step(avg_loss)
        
    return avg_loss, avg_metric_loss


@timer_decorator
def train(net, 
          criterion, 
          optimizer,
          train_loader, 
          val_loader, 
          n_epochs,
          metric_criterion,
          scheduler=None,
          grad_clip_val=0):
    print("===Start training===")
    for epoch in range(n_epochs):
        net, train_loss = net_train(net=net,
                        optimizer=optimizer,
                        criterion=criterion,
                        dataloader=train_loader,
                        grad_clip_val=grad_clip_val)

        val_loss, val_metric_loss = net_eval(net=net,
                                             dataloader=val_loader,
                                             criterion=criterion,
                                             metric_criterion=metric_criterion,
                                             scheduler=scheduler)
        
        if epoch % 1 == 0 or epoch == n_epochs - 1:
            if scheduler:
                print(f"Epoch {epoch:04d} — train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, val_NMSE: {val_metric_loss:.2f}, lr: {scheduler.get_last_lr()[0]}")
            else:
                print(f"Epoch {epoch:04d} — train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, val_NMSE: {val_metric_loss:.2f}")

    print("===Training complete===")


def net_inference(net, x, deterministic=None):
    net = net.eval()
    with torch.no_grad():
        if deterministic:
            y = net(x, deterministic=deterministic)
        else:
            y = net(x)
    return y
