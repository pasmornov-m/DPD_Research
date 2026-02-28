import os
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from modules import metrics, utils, data_loader


class ILCSignal:
    file_path = "ilc_signals"
    def __init__(self, pa_model, epochs=100, learning_rate=0.1):
        self.pa_model = pa_model
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.uk_train = None
        self.uk_val = None
    
    @staticmethod
    def load_signal(file_path: str = file_path):
        base_path = Path(file_path)
        train_path = base_path / "uk_train.csv"
        val_path = base_path / "uk_val.csv"

        if not train_path.exists():
            raise FileNotFoundError(f"{train_path} not found")
        if not val_path.exists():
            raise FileNotFoundError(f"{val_path} not found")

        uk_train = data_loader.load_csv_to_tensor(str(train_path))
        uk_val = data_loader.load_csv_to_tensor(str(val_path))

        return uk_train, uk_val

    def optimize(self, container):
        self.uk_train = ilc_signal(container.train_input, 
                                   container.train_output_target, 
                                   self.pa_model, 
                                   self.epochs, 
                                   self.learning_rate)
        self.uk_val = ilc_signal(container.val_input,
                                 container.val_output_target, 
                                 self.pa_model, 
                                 self.epochs, 
                                 self.learning_rate)
    
    def get_signals(self):
        return self.uk_train, self.uk_val
    
    @staticmethod
    def _tensor_to_dataframe(signal: torch.Tensor) -> pd.DataFrame:
        signal = signal.detach().cpu()

        if signal.is_complex():
            I = signal.real.numpy()
            Q = signal.imag.numpy()
        else:
            if signal.dim() != 2 or signal.size(1) != 2:
                raise ValueError("Signal must be complex tensor or shape [N,2]")
            I = signal[:, 0].numpy()
            Q = signal[:, 1].numpy()

        return pd.DataFrame({"I": I, "Q": Q})

    def save_signals(self, folder_path: str = file_path):
        os.makedirs(folder_path, exist_ok=True)

        if self.uk_train is not None:
            df_train = self._tensor_to_dataframe(self.uk_train)
            df_train.to_csv(f"{folder_path}/uk_train.csv", index=False)

        if self.uk_val is not None:
            df_val = self._tensor_to_dataframe(self.uk_val)
            df_val.to_csv(f"{folder_path}/uk_val.csv", index=False)

@utils.timer_decorator
def ilc_signal(input_data, target_data, pa_model, epochs=100, learning_rate=0.1):
    u = torch.nn.Parameter(input_data.clone(), requires_grad=True)
    optimizer = torch.optim.Adam([u], lr=learning_rate)

    pa_model = pa_model.eval()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pa_output = pa_model.forward(u)
        loss = metrics.compute_mse(pa_output, target_data)
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
    return net, losses


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


@utils.timer_decorator
def train(net, 
          criterion, 
          optimizer,
          train_loader, 
          val_loader, 
          n_epochs,
          metric_criterion,
          scheduler=None,
          grad_clip_val=1.0):
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
                print(f"Epoch {epoch:04d} — train_loss: {train_loss:.8f}, val_loss: {val_loss:.8f}, val_NMSE: {val_metric_loss:.2f}, lr: {scheduler.get_last_lr()[0]}")
            else:
                print(f"Epoch {epoch:04d} — train_loss: {train_loss:.8f}, val_loss: {val_loss:.8f}, val_NMSE: {val_metric_loss:.2f}")

    print("===Training complete===")
    

class ModelTrainer:
    def __init__(self,
                 net, 
                 optimizer,
                 train_loader, 
                 val_loader,
                 criterion, 
                 metric_criterion,
                 grad_clip_val: int = 0,
                 scheduler=None):
        
        self.net = net
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.metric_criterion = metric_criterion
        self.scheduler = scheduler
        self.grad_clip_val = grad_clip_val
    
    @utils.timer_decorator
    def train(self, epochs: int):
        print("===Start training===")
        for epoch in range(epochs):
            train_loss = self._net_train()
            val_loss, val_metric_loss = self._net_eval()
            self._print_log(epoch, train_loss, val_loss, val_metric_loss)
        print("===Training complete===")
    
    def _print_log(self, epoch, train_loss, val_loss, val_metric_loss):
        if self.scheduler:
            print(f"Epoch {epoch:04d} — train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, val_NMSE: {val_metric_loss:.2f}, lr: {self.scheduler.get_last_lr()[0]}")
        else:
            print(f"Epoch {epoch:04d} — train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, val_NMSE: {val_metric_loss:.2f}")
            
    def _net_train(self):
        self.net.train()
        losses = 0
        for features, targets in self.train_loader:
            self.optimizer.zero_grad()
            out = self.net(features)
            loss = self.criterion(out, targets)
            loss.backward()
            if self.grad_clip_val != 0:
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_val)
            self.optimizer.step()
            losses += loss.item() * features.size(0)
        losses /= len(self.train_loader.dataset)
        return losses
    
    def _net_eval(self):
        self.net.eval()
        with torch.no_grad():
            val_loss = 0
            metric_val_loss = 0
            for features, targets in self.val_loader:
                outputs = self.net(features)
                loss = self.criterion(outputs, targets)
                metric_loss = self.metric_criterion(outputs, targets)
                val_loss += loss.item() * features.size(0)
                metric_val_loss += metric_loss.item() * features.size(0)
        avg_loss = val_loss / len(self.val_loader.dataset)
        avg_metric_loss = metric_val_loss / len(self.val_loader.dataset)
        
        if self.scheduler:
            self.scheduler.step(avg_loss)
            
        return avg_loss, avg_metric_loss


def net_inference(net, x, model_type=None):
    is_complex = x.is_complex()

    if is_complex:
        x = utils.complex_to_iq(x)
    
    if x.dim() == 2:
        if model_type != "LEANN":
            x = x.unsqueeze(0)

    net = net.eval()
    with torch.no_grad():
        result = net(x)
    
    if result is None:
        return
    if is_complex:
        result = torch.squeeze(result)

    result = utils.iq_to_complex(result)
        
    return result
