import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, dropout=0.2):
        super(TCNNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                               padding=(kernel_size - 1) * dilation)
        self.chomp1 = Chomp1d((kernel_size - 1) * dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                               padding=(kernel_size - 1) * dilation)
        self.chomp2 = Chomp1d((kernel_size - 1) * dilation)

        self.network = nn.Sequential(
            self.conv1, self.chomp1, self.relu, self.dropout,
            self.conv2, self.chomp2, self.relu, self.dropout
        )

        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.relu(self.network(x) + self.residual(x))


class TCNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_channels, kernel_size=3, dropout=0.2):
        super(TCNNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(TCNNBlock(in_channels, out_channels, kernel_size, dilation=dilation_size, dropout=dropout))

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change to (batch_size, channels, seq_length)
        out = self.network(x)
        return self.fc(out[:, :, -1])  # Extract last time step


def train_tcnn(model, train_loader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for seq, label in progress_bar:
            seq, label = seq.to('cuda'), label.to('cuda')
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output.squeeze(), label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            progress_bar.set_postfix({"Loss": f"{total_loss:.4f}"})

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {total_loss:.4f}")


def evaluate_tcnn(model, test_loader):
    model.eval()
    actuals, predictions = [], []

    with torch.no_grad():
        for seq, label in test_loader:
            seq, label = seq.to('cuda'), label.to('cuda')
            output = model(seq)
            predictions.extend(output.cpu().numpy())
            actuals.extend(label.cpu().numpy())

    test_r2 = r2_score(actuals, predictions)
    test_rmse = mean_squared_error(actuals, predictions, squared=False)

    print(f"Final Model | Test R²: {test_r2:.4f} | Test RMSE: {test_rmse:.4f}")
