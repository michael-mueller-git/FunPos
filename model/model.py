import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import numpy as np

from model.convlstm import ConvLSTM

class Flatten(torch.nn.Module):
    def forward(self, input):
        b, seq_len, _, h, w = input.size()
        return input.view(b, seq_len, -1)

def gaussian(x, mean=0, std=1):
    return torch.exp((-(x - mean) ** 2)/(2* std ** 2))

class FunPos_Model(nn.Module):
    def __init__(self):
        super().__init__()

        reduce = 1
        self.conv1 = nn.Conv2d(
                in_channels=config.img_channels,
                out_channels=16,
                kernel_size=3,
                stride=1
        )

        reduce += 1
        self.conv2 = nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1
        )

        # reduce += 1
        # self.conv3 = nn.Conv2d(
        #         in_channels=32,
        #         out_channels=64,
        #         kernel_size=3,
        #         stride=1
        # )

        # reduce += 1
        self.pool1 = nn.MaxPool2d(2, 2)
        # self.conv4 = nn.Conv2d(
        #         in_channels=64,
        #         out_channels=64,
        #         kernel_size=3,
        #         stride=1
        # )

        self.convlstm1 = ConvLSTM(
                img_size=(int(config.img_height/2-reduce), int(config.img_width/2-reduce)),
                input_dim=32,
                hidden_dim=config.convlstm_hidden_dim*2,
                kernel_size=(3,3),
                cnn_dropout=0.1,
                rnn_dropout=0.1,
                batch_first=True,
                bias=False,
                layer_norm=True,
                return_sequence=True,
                bidirectional=True
        )

        self.convlstm2 = ConvLSTM(
                img_size=(int(config.img_height/2-reduce), int(config.img_width/2-reduce)),
                input_dim=256,
                hidden_dim=config.convlstm_hidden_dim*2,
                kernel_size=(3,3),
                cnn_dropout=0.1,
                rnn_dropout=0.1,
                batch_first=True,
                bias=False,
                layer_norm=False,
                return_sequence=True,
                bidirectional=True
        )

        self.flatten = Flatten()

        self.fc1 = nn.Linear(int(config.img_width-(2*reduce))*int(config.img_height-(2*reduce))*config.convlstm_hidden_dim, 128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, x, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()
        x_new = []
        for t in range(config.seq_len):
            a = self.conv1(x[:,t,:,:,:])
            a = self.conv2(a)
            a = self.pool1(a)
            # a = self.conv3(a)
            x_new.append(a)
        x = torch.stack(x_new, dim=1)

        x, last_state, last_state_inv = self.convlstm1(x)

        if False:
            x_new = []
            for t in range(config.seq_len):
                x_new.append( self.pool2(x[:,t,:,:,:]) )
            x = torch.stack(x_new, dim=1)

        x, last_state, last_state_inv = self.convlstm2(x)

        # x = torch.flatten(x)
        x = self.flatten(x)
        # x = torch.sigmoid(x)
        x = self.fc1(x)
        # x = torch.sigmoid(x)
        x = self.fc2(x)
        # x = torch.sigmoid(x)
        # x = gaussian(x)

        return x
