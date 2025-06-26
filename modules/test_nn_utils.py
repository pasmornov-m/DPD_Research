import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from modules.metrics import calculate_gain_complex


def load_dataset(path_dataset):
    X_train = pd.read_csv(os.path.join(path_dataset, 'train_input.csv')).to_numpy()
    y_train = pd.read_csv(os.path.join(path_dataset, 'train_output.csv')).to_numpy()
    X_val = pd.read_csv(os.path.join(path_dataset, 'val_input.csv')).to_numpy()
    y_val = pd.read_csv(os.path.join(path_dataset, 'val_output.csv')).to_numpy()
    X_test = pd.read_csv(os.path.join(path_dataset, 'test_input.csv')).to_numpy()
    y_test = pd.read_csv(os.path.join(path_dataset, 'test_output.csv')).to_numpy()
    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_segments(args):
    """
    Split the IQ_data into segments of size nperseg. Zero padding is applied
    if the last section is not of length nperseg.
    """
    nperseg = args.nperseg
    path_dataset = os.path.join('datasets', args.dataset_name)
    train_input = pd.read_csv(os.path.join(path_dataset, 'train_input.csv'))
    train_output = pd.read_csv(os.path.join(path_dataset, 'train_output.csv'))
    val_input = pd.read_csv(os.path.join(path_dataset, 'val_input.csv'))
    val_output = pd.read_csv(os.path.join(path_dataset, 'val_output.csv'))
    test_input = pd.read_csv(os.path.join(path_dataset, 'test_input.csv'))
    test_output = pd.read_csv(os.path.join(path_dataset, 'test_output.csv'))

    def split_segments(IQ_data):
        num_samples = IQ_data.shape[0]
        segments = []
        for i in range(0, num_samples, nperseg):
            segment = IQ_data[i:i + nperseg]
            if segment.shape[0] < nperseg:
                padding_shape = (nperseg - segment.shape[0], 2)
                segment = torch.vstack((segment, torch.zeros(padding_shape)))
            segments.append(segment)
        return np.array(segments)

    train_input_segments = split_segments(train_input)
    train_output_segments = split_segments(train_output)
    val_input_segments = split_segments(val_input)
    val_output_segments = split_segments(val_output)
    test_input_segments = split_segments(test_input)
    test_output_segments = split_segments(test_output)

    return train_input_segments, train_output_segments, val_input_segments, val_output_segments, test_input_segments, test_output_segments


def get_training_frames(segments, seq_len, stride=1):
    """
    For each segment, apply the framing operation.

    Args:
    - segments (3D array): The segments produced by get_ofdm_segments.
    - seq_len (int): The length of each frame.
    - stride_length (int, optional): The step between frames. Default is 1.

    Returns:
    - 3D array where the first dimension is the total number of frames,
      the second dimension is seq_len, and the third dimension is 2 (I and Q).
    """

    all_frames = []
    for segment in segments:
        num_frames = (segment.shape[0] - seq_len) // stride + 1
        for i in range(num_frames):
            frame = segment[i * stride: i * stride + seq_len]
            all_frames.append(frame)

    return np.array(all_frames)


class IQSegmentDataset(Dataset):
    def __init__(self, features, targets, nperseg=16384):
        self.nperseg = nperseg

        features = self.split_segments(features)
        targets = self.split_segments(targets)
        self.features = torch.Tensor(features)
        self.targets = torch.Tensor(targets)

    def split_segments(self, sequence):
        num_samples = len(sequence)
        segments = []
        for i in range(0, num_samples, self.nperseg):
            segment = sequence[i:i + self.nperseg]
            if len(segment) < self.nperseg:
                padding_shape = (self.nperseg - segment.shape[0], 2)
                segment = torch.vstack((segment, torch.zeros(padding_shape)))
            segments.append(segment)
        return np.array(segments)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx, ...]
        targets = self.targets[idx, ...]
        return features, targets


class IQFrameDataset(Dataset):
    def __init__(self, features, targets, frame_length, stride=1):
        # Convert segments into frames
        self.features = torch.Tensor(self.get_frames(features, frame_length, stride))
        self.targets = torch.Tensor(self.get_frames(targets, frame_length, stride))

    @staticmethod
    def get_frames(sequence, frame_length, stride_length):
            frames = []
            sequence_length = len(sequence)
            num_frames = (sequence_length - frame_length) // stride_length + 1
            for i in range(num_frames):
                frame = sequence[i * stride_length: i * stride_length + frame_length]
                frames.append(frame)
            return np.stack(frames)
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def data_prepare(X, y, frame_length, degree):
    Input = []
    Output = []
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    for k in range(X.shape[0]):
        Complex_In = torch.complex(X[k, :, 0], X[k, :, 1])
        Complex_Out = torch.complex(y[k, :, 0], y[k, :, 1])
        ulength = len(Complex_In) - frame_length
        Input_matrix = torch.complex(torch.zeros(ulength, frame_length),
                                     torch.zeros(ulength, frame_length))
        degree_matrix = torch.complex(torch.zeros(ulength - frame_length, frame_length * frame_length * degree),
                                      torch.zeros(ulength - frame_length, frame_length * frame_length * degree))
        for i in range(ulength):
            Input_matrix[i, :] = Complex_In[i:i + frame_length]
        for j in range(1, degree):
            for h in range(frame_length):
                degree_matrix[:,
                (j - 1) * frame_length * frame_length + h * frame_length:(j - 1) * frame_length * frame_length + (
                        h + 1) * frame_length] = Input_matrix[:ulength - frame_length] * torch.pow(
                    abs(Input_matrix[h:h + ulength - frame_length, :]), j)
        Input_matrix = torch.cat((Input_matrix[:ulength - frame_length], degree_matrix), dim=1)
        b_output = np.array(Complex_Out[:len(Complex_In) - 2 * frame_length])
        b_input = np.array(Input_matrix)
        Input.append(b_input)
        Output.append(b_output)

    return Input, Output


class IQFrameDataset_gmp(Dataset):
    def __init__(self, segment_dataset, frame_length, degree, stride_length=1):
        """
        Initialize the frame dataset using a subset of IQSegmentDataset.

        Args:
        - segment_dataset (IQSegmentDataset): The pre-split segment dataset.
        - seq_len (int): The length of each frame.
        - stride_length (int, optional): The step between frames. Default is 1.
        """

        # Extract segments from the segment_dataset
        IQ_in_segments = [item[0] for item in segment_dataset]
        IQ_out_segments = [item[1] for item in segment_dataset]

        # Convert the list of tensors to numpy arrays
        IQ_in_segments = torch.stack(IQ_in_segments).numpy()
        IQ_out_segments = torch.stack(IQ_out_segments).numpy()

        self.IQ_in_frames, self.IQ_out_frames = data_prepare(IQ_in_segments, IQ_out_segments, frame_length, degree)

    def __len__(self):
        return len(self.IQ_in_frames)

    def __getitem__(self, idx):
        return self.IQ_in_frames[idx], self.IQ_out_frames[idx]


#------------------------------------------------------------------------------------------------------------------------------------------

def build_dataloaders(path_dataset, frame_length, batch_size, batch_size_eval, nperseg, step=None):
    """step == 'train_dpd' or nothing"""
    
    # Load Dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(path_dataset=path_dataset)

    # Apply the PA Gain if training DPD
    target_gain = calculate_gain_complex(X_train, y_train)

    if step == 'train_dpd':
        y_train = target_gain * X_train
        y_val = target_gain * X_val
        y_test = target_gain * X_test

    # Extract Features
    input_size = X_train.shape[-1]

    # Define PyTorch Datasets
    train_set = IQFrameDataset(X_train, y_train, frame_length=frame_length)
    val_set = IQSegmentDataset(X_val, y_val, nperseg=nperseg)
    test_set = IQSegmentDataset(X_test, y_test, nperseg=nperseg)

    # Define PyTorch Dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size_eval, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size_eval, shuffle=False)

    return (train_loader, val_loader, test_loader), input_size

#------------------------------------------------------------------------------------------------------------------------------------



def net_train(net,
              dataloader,
              optimizer,
              criterion,
              grad_clip_val):
    # Set Network to Training Mode
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
    loss = np.mean(losses)
    return net, loss


def net_eval(net, dataloader, criterion, metric_criterion=None):
    net = net.eval()
    with torch.no_grad():
        losses = []
        metric_losses =[]
        prediction = []
        ground_truth = []
        for features, targets in dataloader:
            outputs = net(features)
            loss = criterion(outputs, targets)
            if metric_criterion:
                metric_loss = metric_criterion(outputs, targets)
            prediction.append(outputs)
            ground_truth.append(targets)
            losses.append(loss.item())
            metric_losses.append(metric_loss)
    avg_loss = np.mean(losses)
    avg_metric_loss = np.mean(metric_losses)
    prediction = torch.cat(prediction, dim=0).numpy()
    ground_truth = torch.cat(ground_truth, dim=0).numpy()
    return net, prediction, ground_truth, avg_loss, avg_metric_loss





def train(net, 
          criterion, 
          optimizer,
          train_loader, 
          val_loader, 
          grad_clip_val,
          n_epochs,
          metric_criterion):
    print("Starting training...")
    for epoch in range(n_epochs):
        # -----------
        # Train
        # -----------
        net, train_loss = net_train(net=net,
                        optimizer=optimizer,
                        criterion=criterion,
                        dataloader=train_loader,
                        grad_clip_val=grad_clip_val)

        # -----------
        # Validation
        # -----------
        _, prediction, ground_truth, val_loss, val_metric_loss = net_eval(net=net,
                                               criterion=criterion,
                                               dataloader=val_loader,
                                               metric_criterion=metric_criterion)

        print(f"Epoch {epoch:02d} — train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, val_NMSE: {val_metric_loss:.2f}")

    print("Training Completed...")
    print(" ")



def run_nn_model(net, x):
    net = net.eval()
    with torch.no_grad():
        x = x.unsqueeze(0)
        y = net(x)
        y = torch.squeeze(y)
    return y