import torch
from torch import nn
from modules.metrics import compute_mse
import time
from datetime import timedelta


def ilc_signal(input_data, target_data, pa_model, epochs=1000000, learning_rate=0.1):
    u = torch.nn.Parameter(input_data.clone(), requires_grad=True)
    optimizer = torch.optim.Adam([u], lr=learning_rate)
    with torch.no_grad():
        for param in pa_model.parameters():
            param.requires_grad = False
        

    pa_model = pa_model.eval()
    start = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pa_output = pa_model.forward(u)
        loss = compute_mse(pa_output, target_data)
        loss.backward()
        optimizer.step()

        if epoch%100==0 or epoch == epochs - 1:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")
    elapsed = time.time() - start
    print(f"Время расчёта ilc_signal: {timedelta(seconds=round(elapsed))}")
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
    start = time.time()
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
        
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:04d} — train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, val_NMSE: {val_metric_loss:.2f}")

    elapsed = time.time() - start
    print(f"Время обучения: {timedelta(seconds=round(elapsed))}")
    print("===Training complete===")


def net_inference(net, x):
    net = net.eval()
    with torch.no_grad():
        y = net(x)
    return y
