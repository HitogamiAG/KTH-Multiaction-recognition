import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, num_classes: int, input_size: int, num_layers: int, seq_length: int) -> None:
        """Initiale parameters of the model

        Args:
            num_classes (int): Number of classes
            input_size (int): Number of elements in one record
            num_layers (int): Number of layers in LSTM
            seq_length (int): Number of records in one input example
        """
        super(LSTMModel, self).__init__()
        self.labels = ['boxing', 'jogging', 'running',
                       'walking', 'handclapping', 'handwaving']
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.lstm_1_size = 16 * 1
        self.lstm_2_size = 32 * 1
        self.lstm_3_size = 16 * 1
        self.seq_length = seq_length

        self.lstm_1 = nn.LSTM(input_size=input_size, hidden_size=self.lstm_1_size,
                              num_layers=num_layers, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=self.lstm_1_size, hidden_size=self.lstm_2_size,
                              num_layers=num_layers, batch_first=True)
        self.lstm_3 = nn.LSTM(input_size=self.lstm_2_size, hidden_size=self.lstm_3_size,
                              num_layers=num_layers, batch_first=True)

        self.fc_1 = nn.Sequential(nn.Linear(self.lstm_3_size, 128))
        self.fc_2 = nn.Sequential(nn.Linear(128, 32))
        self.classifier = nn.Sequential(nn.Linear(32, self.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through model

        Args:
            x (torch.Tensor): Input

        Returns:
            torch.Tensor: Output
        """
        h_1 = torch.zeros(self.num_layers, x.size(0), self.lstm_1_size)
        c_1 = torch.zeros(self.num_layers, x.size(0), self.lstm_1_size)

        h_2 = torch.zeros(self.num_layers, x.size(0), self.lstm_2_size)
        c_2 = torch.zeros(self.num_layers, x.size(0), self.lstm_2_size)

        h_3 = torch.zeros(self.num_layers, x.size(0), self.lstm_3_size)
        c_3 = torch.zeros(self.num_layers, x.size(0), self.lstm_3_size)

        output, (h_1, c_1) = self.lstm_1(x, (h_1, c_1))

        output, (h_2, c_2) = self.lstm_2(output, (h_2, c_2))

        output, (h_3, c_3) = self.lstm_3(output, (h_3, c_3))

        hn = h_3.view(-1, self.lstm_3_size)
        out = self.fc_1(hn)
        out = self.fc_2(out)
        out = self.classifier(out)

        return out
