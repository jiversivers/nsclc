import numpy as np
import torch
from torch import nn


# region MLPs
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


class ParallelMLPNet(nn.Module):
    def __init__(self, input_size):
        super(ParallelMLPNet, self).__init__()
        self.name = 'Parallel MLP'
        self.input_size = input_size
        self.dims = self.input_size[0]
        self.input_nodes = np.prod(self.input_size[1:])
        self.flat = nn.Flatten()
        self.fc1 = nn.ModuleList([nn.Linear(self.input_nodes, 64) for _ in range(self.dims)])
        self.fc2 = nn.ModuleList([nn.Linear(64, 128) for _ in range(self.dims)])
        self.out1 = nn.ModuleList([nn.Linear(128, 1) for _ in range(self.dims)])

        self.fc3 = nn.Linear(self.dims, 64)
        self.out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x[torch.isnan(x)] = 0
        xii = torch.tensor([], device=x.device)
        # Create independent MLP for each layer of the input
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
        self.input_nodes = np.prod(self.input_size[1:])
        self.flat = nn.Flatten()
        self.bn = nn.ModuleList([nn.BatchNorm1d(self.input_nodes) for _ in range(self.dims)])
        self.fc1 = nn.ModuleList([nn.Linear(self.input_nodes, 64) for _ in range(self.dims)])
        self.fc2 = nn.ModuleList([nn.Linear(64, 128) for _ in range(self.dims)])
        self.fc3 = nn.ModuleList([nn.Linear(128, 64) for _ in range(self.dims)])
        self.out1 = nn.ModuleList([nn.Linear(64, 1) for _ in range(self.dims)])

        self.fc4 = nn.Linear(self.dims, 25)
        self.out = nn.Linear(25, 1)

        self.drop = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x[torch.isnan(x)] = 0
        xii = torch.tensor([], device=x.device)
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


# endregion

# region RNNS
class RNNet(nn.Module):
    def __init__(self, input_size):
        super(RNNet, self).__init__()
        self.name = 'RN Net'
        self.input_size = input_size
        self.hidden_size = 32
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
        self.hidden_size = 32
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


class ParallelRNNet(nn.Module):
    def __init__(self, input_size):
        super(ParallelRNNet, self).__init__()
        self.name = 'Parallel RN Net'
        self.input_size = input_size
        self.dims = self.input_size[0]
        self.hidden_size = 16
        self.output_size = 1
        self.num_layers = 2

        self.rnn = nn.ModuleList(
            [nn.RNN(input_size=1, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True) for _ in
             range(self.dims)])
        self.fc1 = nn.Linear(self.hidden_size * self.dims, 64)
        self.fc2 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x[torch.isnan(x)] = 0
        xii = torch.tensor([], device=x.device)

        for ii in range(self.dims):
            xi = x[:, ii].unsqueeze(2)
            xi, _ = self.rnn[ii](xi)
            xi = xi[:, -1, :]
            xii = torch.cat((xii, xi), 1)

        x = self.fc1(xii)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigm(x)
        return x


class RegularizedParallelRNNet(nn.Module):
    def __init__(self, input_size):
        super(RegularizedParallelRNNet, self).__init__()
        self.name = 'Regularaized Parallel RN Net'
        self.input_size = input_size
        self.dims = self.input_size[0]
        self.hidden_size = 16
        self.output_size = 1
        self.num_layers = 2

        self.bn = nn.ModuleList([nn.BatchNorm1d(1) for _ in range(self.dims)])
        self.rnn = nn.ModuleList([nn.RNN(input_size=1, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                         batch_first=True, dropout=0.25) for _ in range(self.dims)])
        self.fc1 = nn.Linear(self.hidden_size * self.dims, 64)
        self.fc2 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x[torch.isnan(x)] = 0
        xii = torch.tensor([], device=x.device)

        for ii in range(self.dims):
            xi = x[:, ii].unsqueeze(2)
            xi, _ = self.rnn[ii](xi)
            xi = xi[:, -1, :]
            xii = torch.cat((xii, xi), 1)

        x = self.fc1(xii)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigm(x)
        return x


# endregion

# region CNNs
class CNNet(nn.Module):
    def __init__(self, input_size):
        super(CNNet, self).__init__()
        self.name = 'CN Net'
        self.input_size = input_size

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
        self.input_size = input_size

        self.bn = nn.BatchNorm2d(input_size[0])
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

        self.conv1 = nn.ModuleList([nn.Conv2d(1, 32, kernel_size=3, padding=1) for _ in range(self.dims)])
        self.conv2 = nn.ModuleList([nn.Conv2d(32, 64, kernel_size=3, padding=1) for _ in range(self.dims)])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat = nn.Flatten()
        self.fc1 = nn.ModuleList([nn.Linear(64 * int(np.prod(input_size[1:]) // 16), 128) for _ in range(self.dims)])
        self.fc2 = nn.ModuleList([nn.Linear(128, 1) for _ in range(self.dims)])
        self.fc3 = nn.Linear(self.dims, 1)

        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x[torch.isnan(x)] = 0
        xii = torch.tensor([], device=x.device)
        # Forward pass of CNN for each layer
        for ii in range(self.dims):
            xi = x[:, ii].unsqueeze(1)
            xi = self.conv1[ii](xi).to(x.device)
            xi = self.pool(xi)
            xi = self.conv2[ii](xi)
            xi = self.pool(xi)
            xi = self.flat(xi)
            xi = self.fc1[ii](xi)
            xi = self.relu(xi)
            xi = self.fc2[ii](xi)
            xii = torch.cat((xii, xi), 1)

        # Recombine layers into final dense layer
        x = self.fc3(xii)
        x = self.sigm(x)
        return x


class RegularizedParallelCNNet(nn.Module):
    def __init__(self, input_size):
        super(RegularizedParallelCNNet, self).__init__()
        self.name = 'Regularized Parallel CN Net'
        self.input_size = input_size
        self.dims = self.input_size[0]

        self.bn = nn.ModuleList([nn.BatchNorm2d(1) for _ in range(self.dims)])
        self.conv1 = nn.ModuleList([nn.Conv2d(1, 32, kernel_size=3, padding=1) for _ in range(self.dims)])
        self.conv2 = nn.ModuleList([nn.Conv2d(32, 64, kernel_size=3, padding=1) for _ in range(self.dims)])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat = nn.Flatten()
        self.fc1 = nn.ModuleList([nn.Linear(64 * int(np.prod(input_size[1:]) // 16), 128) for _ in range(self.dims)])
        self.fc2 = nn.ModuleList([nn.Linear(128, 1) for _ in range(self.dims)])
        self.fc3 = nn.Linear(self.dims, 1)

        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        x[torch.isnan(x)] = 0
        xii = torch.tensor([], device=x.device)
        # Forward pass of CNN for each layer
        for ii in range(self.dims):
            xi = x[:, ii].unsqueeze(1)
            xi = self.bn[ii](xi)
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

        # Recombine layers into final dense layer
        x = self.fc3(xii)
        x = self.sigm(x)
        return x


# endregion

# region More Involved Classifiers
class RegularizedMLPNetWithPretrainedFeatureExtractor(nn.Module):
    def __init__(self, input_size, feature_extractor, layer=''):
        super(RegularizedMLPNetWithPretrainedFeatureExtractor, self).__init__()
        self.feature_extractor = feature_extractor
        self.input_size = input_size
        self.layer = layer

        self.feature_map_dims = self.get_features(
            torch.rand(1, *input_size, device=next(feature_extractor.parameters()).device)).shape

        # get = {}
        #
        # def get_features():
        #     # Create hook for feature extraction
        #     def hook(model, input, output):
        #         get['features'] = output.detach()
        #
        #     return hook
        #
        # eval(f'self.feature_extractor{layer}.register_forward_hook(get_features())')
        #
        # dry_run = feature_extractor(
        #     torch.rand(1, *input_size, device=next(feature_extractor.parameters()).device)).shape
        # print(dry_run)
        # print(get['features'].shape)

        # Get average value for each feature map
        self.GlobalAvgPool = nn.AvgPool2d(self.feature_map_dims[-2::], stride=2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(self.feature_map_dims[1], 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        # Force input to match device and store original to put it back from where it came
        input_device = x.device

        # Extract features
        x = x.to(next(self.feature_extractor.parameters()).device)
        x = self.get_features(x)

        # Classify
        x = x.to(next(self.parameters()).device)
        x = self.GlobalAvgPool(x)
        x = self.flat(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x.to(input_device)

    def get_features(self, x):
        get = {}

        # Create hook for feature extraction
        def hook(model, input, output):
            get['features'] = output.detach()

        fh = eval(f'self.feature_extractor{self.layer}.register_forward_hook(hook)')

        # Get the features from the specified layer via the hook
        _ = self.feature_extractor(x)

        fh.remove() # Clean up the hook
        return get['features']
# endregion
