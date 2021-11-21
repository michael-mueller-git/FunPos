import torch
import torch.nn as nn
from utils.config import CONFIG

from modules.convlstm import ConvLSTM

MODEL='model1'

class Flatten(torch.nn.Module):
    def forward(self, input):
        b, seq_len, _, h, w = input.size()
        return input.view(b, seq_len, -1)


class FunPosModel(nn.Module):
    """ FunPos Model """
    def __init__(self):
        super().__init__()

        reduce = 1
        self.conv1 = nn.Conv2d(
                in_channels = CONFIG[MODEL]['img_channels'],
                out_channels = 16,
                kernel_size = 3,
                stride = 1
        )

        reduce += 1
        self.conv2 = nn.Conv2d(
                in_channels = 16,
                out_channels = 32,
                kernel_size = 3,
                stride = 1
        )

        self.pool1 = nn.MaxPool2d(2, 2)

        self.convlstm1 = ConvLSTM(
                img_size = (int(CONFIG[MODEL]['img_height']/2-reduce), int(CONFIG[MODEL]['img_width']/2-reduce)),
                input_dim = 32,
                hidden_dim = CONFIG[MODEL]['convlstm_hidden_dim']*2,
                kernel_size = (3,3),
                cnn_dropout = 0.1,
                rnn_dropout = 0.1,
                batch_first = True,
                bias = False,
                layer_norm = True,
                return_sequence = True,
                bidirectional = True
        )

        self.convlstm2 = ConvLSTM(
                img_size = (int(CONFIG[MODEL]['img_height']/2-reduce), int(CONFIG[MODEL]['img_width']/2-reduce)),
                input_dim = 256,
                hidden_dim = CONFIG[MODEL]['convlstm_hidden_dim']*2,
                kernel_size = (3,3),
                cnn_dropout = 0.1,
                rnn_dropout = 0.1,
                batch_first = True,
                bias = False,
                layer_norm = False,
                return_sequence = True,
                bidirectional = True
        )

        self.flatten = Flatten()

        self.fc1 = nn.Linear(
                int(CONFIG[MODEL]['img_width']-(2*reduce))*int(CONFIG[MODEL]['img_height']-(2*reduce))*CONFIG[MODEL]['convlstm_hidden_dim'],
                128
        )
        self.fc2 = nn.Linear(128, 1)


    def forward(self, x, hidden_state=None):
        """ Forward pass

        Args:
            x (torch.tensor): 5-D Tensor of shape (batch, time, channel, height, width)

        Returns:
            tensor: prediction
        """

        b, seq_len, _, h, w = x.size()
        x_new = []
        for t in range(CONFIG[MODEL]['seq_len']):
            a = self.conv1(x[:,t,:,:,:])
            a = self.conv2(a)
            a = self.pool1(a)
            x_new.append(a)
        x = torch.stack(x_new, dim=1)

        x, last_state, last_state_inv = self.convlstm1(x)
        x, last_state, last_state_inv = self.convlstm2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
