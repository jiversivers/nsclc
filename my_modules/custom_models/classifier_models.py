import numpy as np
import torch
from torch import nn


class MLPNet(nn.Module):
    def __init__(self, input_size):
        super(MLPNet, self).__init__()
        self.name = 'MLP Net'
        self.input_size = input_size
        self.input_nodes = np.prod(self.input_size)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(self.input_nodes, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x[torch.isnan(x)] = 0
        x = self.flat(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigm(x)
        return x


class RegularizedMLPNet(nn.Module):
    def __init__(self, input_size):
        super(RegularizedMLPNet, self).__init__()
        self.name = 'Regularized MLP Net'
        self.input_size = input_size
        self.input_nodes = np.prod(self.input_size)
        self.flat = nn.Flatten()
        self.bn = nn.BatchNorm1d(self.input_nodes)
        self.fc1 = nn.Linear(self.input_nodes, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

        self.drop = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x[torch.isnan(x)] = 0
        x = self.flat(x)
        x = self.bn(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigm(x)
        return x


class RNNet(nn.Module):
    def __init__(self, input_size):
        super(RNNet, self).__init__()
        self.name = 'RN Net'
        self.input_size = input_size
        self.hidden_size = int(np.sqrt(np.prod(self.input_size)))
        self.output_size = 1
        self.num_layers = 2

        self.rnn = nn.RNN(self.input_size[0], self.hidden_size, num_layers=self.num_layers, batch_first=True,
                          nonlinearity='tanh')
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x[torch.isnan(x)] = 0
        batch_size = x.size(0)
        hn = self.hidden_init(batch_size, x.device)
        x = torch.reshape(x, (batch_size, self.input_size[1], self.input_size[0]))
        x, hn = self.rnn(x, hn)
        x = self.fc(x[:, -1])
        x = self.sigm(x)
        return x

    def hidden_init(self, batch_size, device):
        hn = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return hn


class RegularizedRNNet(nn.Module):
    def __init__(self, input_size):
        super(RegularizedRNNet, self).__init__()
        self.name = 'Regularized RN Net'
        self.input_size = input_size
        self.hidden_size = int(np.sqrt(np.prod(self.input_size)))
        print(self.hidden_size)
        self.output_size = 1
        self.num_layers = 2
        self.bn = nn.BatchNorm1d(input_size[0])
        self.rnn = nn.RNN(self.input_size[0], self.hidden_size, self.num_layers, batch_first=True,
                          nonlinearity='tanh', dropout=0.25)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x[torch.isnan(x)] = 0
        batch_size = x.size(0)
        hn = self.hidden_init(batch_size, x.device)
        x = self.bn(x)
        x = torch.reshape(x, (batch_size, self.input_size[1], self.input_size[0]))
        x, hn = self.rnn(x, hn)
        x = self.fc(x[:, -1])
        x = self.sigm(x)
        return x

    def hidden_init(self, batch_size, device):
        hn = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return hn


class ParallelMLPNet(nn.Module):
    def __init__(self, input_size):
        super(ParallelMLPNet, self).__init__()
        self.name = 'Parallel MLP'
        self.input_size = input_size
        self.dims = self.input_size[0]
        self.input_nodes = np.prod(self.input_size[1:])
        self.flat = nn.Flatten()
        self.fc1 = self.dims * [nn.Linear(self.input_nodes, 64)]
        self.fc2 = self.dims * [nn.Linear(64, 128)]
        self.out1 = self.dims * [nn.Linear(128, 1)]

        self.fc3 = nn.Linear(self.dims, 64)
        self.out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x[torch.isnan(x)] = 0
        xii = torch.tensor([])
        # Create independent networks for each layer of the input
        for ii in range(self.dims):
            xi = self.flat(x[:, ii])
            xi = self.fc1[ii](xi)
            xi = self.relu(xi)
            xi = self.fc2[ii](xi)
            xi = self.relu(xi)
            xi = self.out1[ii](xi)
            xii = torch.cat((xii, xi), 1)

        # Recombine layers in final dense layers
        x = self.fc3(xii)
        x = self.relu(x)
        x = self.out(x)
        x = self.sigm(x)
        return x


class RegularizedParallelMLPNet(nn.Module):
    def __init__(self, input_size):
        super(RegularizedParallelMLPNet, self).__init__()
        self.name = 'Regularized Parallel MLP Net'
        self.input_size = input_size
        self.dims = self.input_size[0]
        self.input_nodes = np.prod(self.input_size[1])
        self.flat = nn.Flatten()
        self.bn = self.dims * [nn.BatchNorm1d(self.input_nodes)]
        self.fc1 = self.dims * [nn.Linear(self.input_nodes, 64)]
        self.fc2 = self.dims * [nn.Linear(64, 128)]
        self.fc3 = self.dims * [nn.Linear(128, 64)]
        self.out1 = self.dims * [nn.Linear(64, 1)]

        self.fc4 = nn.Linear(self.dims, 25)
        self.out = nn.Linear(25, 1)

        self.drop = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x[torch.isnan(x)] = 0
        xii = torch.tensor([])
        # Create independent networks for each layer of the input
        for ii in range(self.dims):
            xi = self.flat(x[:, ii])
            xi = self.bn[ii](xi)
            xi = self.drop(xi)
            xi = self.fc1[ii](xi)
            xi = self.relu(xi)
            xi = self.drop(xi)
            xi = self.fc2[ii](xi)
            xi = self.relu(xi)
            xi = self.drop(xi)
            xi = self.fc3[ii](xi)
            xi = self.relu(xi)
            xi = self.out1[ii](xi)
            xii = torch.cat((xii, xi), 1)

        # Recombine layers in final dense layers
        x = self.drop(xii)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.out(x)
        x = self.sigm(x)
        return x


class CNNet(nn.Module):
    def __init__(self, input_size):
        super(CNNet, self).__init__()
        self.name = 'CN Net'

        self.conv1 = nn.Conv2d(input_size[0], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(64 * int(np.prod(input_size[1:]) // 16), 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x[torch.isnan(x)] = 0
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.sigm(x)
        return x


class RegularizedCNNet(nn.Module):
    def __init__(self, input_size):
        super(RegularizedCNNet, self).__init__()
        self.name = 'Regularized CN Net'

        self.bn = nn.BatchNorm2d(self.input_nodes[0])
        self.conv1 = nn.Conv2d(input_size[0], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(64 * int(np.prod(input_size[1:]) // 16), 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x[torch.isnan(x)] = 0
        x = self.bn(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.sigm(x)
        return x


class ParallelCNNet(nn.Module):

    def __init__(self, input_size):
        super(ParallelCNNet, self).__init__()
        self.name = 'Parallel CN Net'
        self.input_size = input_size
        self.dims = self.input_size[0]

        self.conv1 = self.dims * [nn.Conv2d(self.input_size[0], 32, kernel_size=3, padding=1)]
        self.conv2 = self.dims * [nn.Conv2d(32, 64, kernel_size=3, padding=1)]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat = nn.Flatten()
        self.fc1 = self.dims * [nn.Linear(64 * int(np.prod(input_size[1:]) // 16), 128)]
        self.fc2 = self.dims * [nn.Linear(128, 1)]
        self.fc3 = nn.Linear(self.dims, 1)

        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

        def forward(self, x):
            x[torch.isnan(x)] = 0
            xii = torch.tensor([])
            for ii in range(self.dims):
                xi = self.conv1[ii](x[:, ii])
                xi = self.pool(xi)
                xi = self.conv2[ii](xi)
                xi = self.pool(xi)
                xi = self.flat(xi)
                xi = self.fc1[ii](xi)
                xi = self.relu(xi)
                xi = self.fc2[ii](xi)
                xii = torch.cat((xii, xi), 1)

            # Recombine layes into final dense layer
            x = self.fc3(xii)
            x = self.sigm(x)
            return x


class RegularizedParallelCNNet(nn.Module):
    def __init__(self, input_size):
        super(RegularizedParallelCNNet, self).__init__()
        self.name = 'Regularized Parallel CN Net'
        self.input_size = input_size
        self.dims = self.input_size[0]

        self.bn = self.dims * [nn.BatchNorm2d(self.input_nodes)]
        self.conv1 = self.dims * [nn.Conv2d(self.input_size[0], 32, kernel_size=3, padding=1)]
        self.conv2 = self.dims * [nn.Conv2d(32, 64, kernel_size=3, padding=1)]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat = nn.Flatten()
        self.fc1 = self.dims * [nn.Linear(64 * int(np.prod(input_size[1:]) // 16), 128)]
        self.fc2 = self.dims * [nn.Linear(128, 1)]
        self.fc3 = nn.Linear(self.dims, 1)

        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        x[torch.isnan(x)] = 0
        xii = torch.tensor([])
        for ii in range(self.dims):
            xi = self.bn[ii](x[:, ii])
            xi = self.drop(xi)
            xi = self.conv1[ii](xi)
            xi = self.pool(xi)
            xi = self.drop(xi)
            xi = self.conv2[ii](xi)
            xi = self.pool(xi)
            xi = self.drop(xi)
            xi = self.flat(xi)
            xi = self.drop(xi)
            xi = self.fc1[ii](xi)
            xi = self.relu(xi)
            xi = self.fc2[ii](xi)
            xii = torch.cat((xii, xi), 1)

        # Recombine layes into final dense layer
        x = self.fc3(xii)
        x = self.sigm(x)
        return x


