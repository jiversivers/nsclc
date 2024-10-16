from typing import Iterator, Tuple

import numpy as np
import torch
from pretrainedmodels import inceptionresnetv2, xception
from torch import nn
import warnings
import traceback

from torch.nn import Parameter

from my_modules.custom_models import blocks


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

    def forward(self, x):
        x[torch.isnan(x)] = 0
        x = self.flat(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

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
        x = self.relu(xii)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.out(x)

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
        x = self.relu(xii)
        x = self.drop(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.out(x)

        return x


# endregion

# region RNNs
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

    def forward(self, x):
        x[torch.isnan(x)] = 0
        batch_size = x.size(0)
        hn = self.hidden_init(batch_size, x.device)
        x = torch.reshape(x, (batch_size, self.input_size[1], self.input_size[0]))
        x, hn = self.rnn(x, hn)
        x = self.fc(x[:, -1])

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

    def forward(self, x):
        x[torch.isnan(x)] = 0
        batch_size = x.size(0)
        hn = self.hidden_init(batch_size, x.device)
        x = self.bn(x)
        x = torch.reshape(x, (batch_size, self.input_size[1], self.input_size[0]))
        x, hn = self.rnn(x, hn)
        x = self.fc(x[:, -1])

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
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x[torch.isnan(x)] = 0
        xii = torch.tensor([], device=x.device)

        for ii in range(self.dims):
            xi = self.bn(x[:, ii].unsqueeze(2))
            xi, _ = self.rnn[ii](xi)
            xi = xi[:, -1, :]
            xii = torch.cat((xii, xi), 1)

        x = self.drop(xii)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)

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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x[torch.isnan(x)] = 0
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
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
        self.sigmoid = nn.Sigmoid()

        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x[torch.isnan(x)] = 0
        x = self.bn(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

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
        self.sigmoid = nn.Sigmoid()

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
        x = self.relu(xii)
        x = self.fc3(x)
        x = self.sigmoid(x)

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
        self.sigmoid = nn.Sigmoid()

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
        x = self.relu(xii)
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x


# endregion

# region More Involved Classifiers
class CometClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.name = 'Comet Classifier'
        self.input_size = input_size

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(np.prod(self.input_size), 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x[torch.isnan(x)] = 0
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x


class CometClassifierWithBinaryOutput(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.name = 'Comet Classifier'
        self.input_size = input_size

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(np.prod(self.input_size), 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x[torch.isnan(x)] = 0
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class MPMShallowClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.name = 'MPM Shallow Classifier'
        self.input_size = input_size

        self.fc = nn.Linear(np.prod(self.input_size), 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x[torch.isnan(x)] = 0
        x = self.fc(x)
        x = self.softmax(x)
        return x


class FeatureExtractorToClassifier(nn.Module):
    def __init__(self, input_size, feature_extractor, feature_extractor_channels=3, classifier=CometClassifier,
                 layer=None):
        super().__init__()
        self.input_size = input_size
        self.expand_to_rgb = (self.input_size[0] != feature_extractor_channels)

        # Parse input layer
        if layer is not None:
            self.layer = layer
        else:
            self.layer = list(feature_extractor.named_children())[-1][0]
        # model._layer_for_eval = ''.join(['.' + lay for lay in model.layer])

        # Isolate the feature extractor params through the input layer
        self.feature_extractor_params = nn.Module()
        reached_layer = False
        for name, module in feature_extractor.named_children():
            if name is self.layer:
                reached_layer = True
            elif reached_layer:
                break
            setattr(self.feature_extractor_params, name, module)

        self.exception_list = []

        # Dry run for feature dims
        self.feature_extractor = feature_extractor
        x = torch.rand(1, *self.input_size, device=next(feature_extractor.parameters()).device)
        x = x.expand(-1, 3, -1, -1) if self.expand_to_rgb else x
        self.feature_map_dims = self.get_features(x).shape

        # Get average value for each feature map
        self.global_avg_pool = nn.AvgPool2d(self.feature_map_dims[-2::])
        self.flat = nn.Flatten()

        # Init classifier based on size (channels) of feature maps or use pre-init'd classifier
        if type(classifier) is type:
            self.classifier = classifier((self.feature_map_dims[1]))

            # Put init classifier onto same device as feature extractor
            self.classifier.to(next(self.feature_extractor_params.parameters()).device)
        else:
            self.classifier = classifier
            # Check for device compatibility
            if next(self.classifier.parameters()).device != next(self.feature_extractor_params.parameters()).device:
                warnings.warn('Model classifier and feature extractor appear to be on different devices.',
                              RuntimeWarning)

        self.name = f'{type(feature_extractor).__name__} Features to {type(classifier).__name__} Classifier'

    def forward(self, x):
        # Force input to match device and store original to put it back from where it came
        input_device = x.device
        x.to(next(self.feature_extractor_params.parameters()).device)

        # Extract features with nan-masking as 0
        x[torch.isnan(x)] = 0
        x = x.expand(-1, 3, -1, -1) if self.expand_to_rgb else x
        x = self.get_features(x)

        # Pool and flatten feature maps
        x = self.global_avg_pool(x)
        x = self.flat(x)

        # Classify
        x.to(next(self.classifier.parameters()).device)
        x = self.classifier(x)

        return x.to(input_device)

    def get_features(self, x):
        get = {}

        # Create hook for feature extraction
        def hook(model, input, output):
            get['features'] = output
            return

        # Set hook in feature extractor
        fh = getattr(self.feature_extractor, self.layer).register_forward_hook(hook, always_call=True)
        # fh = eval(f'model.feature_extractor_params{model._layer_for_eval}.register_forward_hook(hook, always_call=True)')

        # Get the features from the specified layer via the hook, even if the image is "too small" for the final avg filter
        try:
            x.to(next(self.feature_extractor_params.parameters()).device)
            _ = self.feature_extractor(x)
        except RuntimeError as e:
            # Print the first occurrence of the error type
            if type(e) not in self.exception_list:
                self.exception_list.append(type(e))
                print(f'{''.join(traceback.format_tb(e.__traceback__))}\n '
                      f'Warning {type(e)} error caught during feature extraction\n'
                      f'{e}')
            else:
                pass

        # Clean up the hook
        fh.remove()
        return get['features']

    def named_parameters(
            self,
            prefix: str = '',
            recurse: bool = True,
            remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        # Add feature extractor params
        for name, param in self.feature_extractor_params.named_parameters(prefix='feature_extractor',
                                                                          recurse=recurse,
                                                                          remove_duplicate=remove_duplicate):
            yield name, param

        # Add classifier parameters
        for name, param in self.classifier.named_parameters(prefix='classifier',
                                                            recurse=recurse, remove_duplicate=remove_duplicate):
            yield name, param

    def to(self, device):
        self.feature_extractor_params.to(device)
        self.global_avg_pool.to(device)
        self.flat.to(device)
        self.classifier.to(device)


class ResNet18NPlaned(nn.Module):
    def __init__(self, input_size, start_width=64, n_classes=1, sigmoid=True):
        super().__init__()
        self.input_size = input_size
        self.planes = self.input_size[0]
        self.start_width = start_width
        self.name = f'{self.planes}-Planed ResNet18'
        self.sigmoid_on = sigmoid

        self.relu = nn.ReLU(inplace=True)

        # Stem
        self.stem = nn.Sequential(nn.Conv2d(self.planes, self.start_width,
                                            kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                                  nn.BatchNorm2d(self.start_width,
                                                 eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.maxpool_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), ceil_mode=False)

        # Block 2, Pass 1
        width = self.start_width
        self.block2_1 = nn.Sequential(blocks.conv3x3_to_outfun(width, width, 'relu', stride=(1, 1), padding=(1, 1)),
                                      blocks.conv3x3_to_outfun(width, width, 'drop', stride=(1, 1), padding=(1, 1)))
        # Pass 2
        self.block2_2 = nn.Sequential(blocks.conv3x3_to_outfun(width, width, 'relu', stride=(1, 1), padding=(1, 1)),
                                      blocks.conv3x3_to_outfun(width, width, 'drop', stride=(1, 1), padding=(1, 1)))

        # Block 3, Pass 1
        self.block3_1 = nn.Sequential(blocks.conv3x3_to_outfun(width, 2 * width, 'relu', stride=(2, 2), padding=(1, 1)),
                                      blocks.conv3x3_to_outfun(2 * width, 2 * width, 'drop', stride=(1, 1),
                                                               padding=(1, 1)))
        self.concat_adj3 = nn.Conv2d(width, 2 * width, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0))
        # Pass 2
        width *= 2
        self.block3_2 = nn.Sequential(blocks.conv3x3_to_outfun(width, width, 'relu', stride=(1, 1), padding=(1, 1)),
                                      blocks.conv3x3_to_outfun(width, width, 'drop', stride=(1, 1), padding=(1, 1)))

        # Block 4, Pass 1
        self.block4_1 = nn.Sequential(blocks.conv3x3_to_outfun(width, 2 * width, 'relu', stride=(2, 2), padding=(1, 1)),
                                      blocks.conv3x3_to_outfun(2 * width, 2 * width, 'drop', stride=(1, 1),
                                                               padding=(1, 1)))
        self.concat_adj4 = nn.Conv2d(width, 2 * width, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0))
        # Pass 2
        width *= 2
        self.block4_2 = nn.Sequential(blocks.conv3x3_to_outfun(width, width, 'relu', stride=(1, 1), padding=(1, 1)),
                                      blocks.conv3x3_to_outfun(width, width, 'drop', stride=(1, 1), padding=(1, 1)))

        # Block 5, Pass 1
        self.block5_1 = nn.Sequential(blocks.conv3x3_to_outfun(width, 2 * width, 'relu', stride=(2, 2), padding=(1, 1)),
                                      blocks.conv3x3_to_outfun(2 * width, 2 * width, 'drop', stride=(1, 1),
                                                               padding=(1, 1)))
        self.concat_adj5 = nn.Conv2d(width, 2 * width, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0))
        # Pass 2
        width *= 2
        self.block5_2 = nn.Sequential(blocks.conv3x3_to_outfun(width, width, 'relu', stride=(1, 1), padding=(1, 1)),
                                      blocks.conv3x3_to_outfun(width, width, 'drop', stride=(1, 1), padding=(1, 1)))

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.fc = nn.Linear(width, 1000)
        self.out = nn.Linear(1000, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Prep
        x[torch.isnan(x)] = 0

        # Stem (Block 1)
        x = self.stem(x)
        x = self.relu(x)
        rx = self.maxpool_1(x)

        # Block 2
        x = self.block2_1(rx)
        rx = self.relu(rx + x)
        x = self.block2_2(rx)
        rx = self.relu(rx + x)

        # Block 3
        x = self.block3_1(rx)
        rx = self.concat_adj3(rx)
        rx = self.relu(rx + x)
        x = self.block3_2(rx)
        rx = self.relu(rx + x)

        # Block 4
        x = self.block4_1(rx)
        rx = self.concat_adj4(rx)
        rx = self.relu(rx + x)
        x = self.block4_2(rx)
        rx = self.relu(rx + x)

        # Block 5
        x = self.block5_1(rx)
        rx = self.concat_adj5(rx)
        rx = self.relu(rx + x)
        x = self.block5_2(rx)
        x = self.relu(rx + x)

        # Head
        x = self.avgpool(x)
        x = self.flat(x)
        x = self.fc(x)
        x = self.out(x)
        x = self.sigmoid(x) if self.sigmoid_on else x
        return x


class AdaptedInputInceptionResnetv2(nn.Module):
    def __init__(self, input_size, start_width=32, sigmoid=True, num_classes=1000, pretrained=False):
        super().__init__()
        self.input_size = input_size
        self.planes = self.input_size[0]
        self.start_width = start_width
        self.name = f'InceptionResNetV2 with {self.planes}-Planed Stem'
        self.sigmoid_on = sigmoid
        self.sigmoid = nn.Sigmoid()

        self.model = inceptionresnetv2(num_classes=num_classes, pretrained=pretrained)
        setattr(self.model.conv2d_1a, 'conv', nn.Conv2d(in_channels=self.planes,
                                                        out_channels=self.start_width,
                                                        kernel_size=self.model.conv2d_1a.conv.kernel_size,
                                                        stride=self.model.conv2d_1a.conv.stride,
                                                        padding=self.model.conv2d_1a.conv.padding,
                                                        dilation=self.model.conv2d_1a.conv.dilation,
                                                        groups=self.model.conv2d_1a.conv.groups,
                                                        bias=self.model.conv2d_1a.conv.bias is not None))
        setattr(self.model.conv2d_1a, 'bn', nn.BatchNorm2d(num_features=self.start_width,
                                                           eps=self.model.conv2d_1a.bn.eps,
                                                           momentum=self.model.conv2d_1a.bn.momentum,
                                                           affine=self.model.conv2d_1a.bn.affine,
                                                           track_running_stats=self.model.conv2d_1a.bn.track_running_stats))
        setattr(self.model.conv2d_2a, 'conv', nn.Conv2d(in_channels=self.start_width,
                                                        out_channels=self.start_width,
                                                        kernel_size=self.model.conv2d_2a.conv.kernel_size,
                                                        stride=self.model.conv2d_2a.conv.stride,
                                                        padding=self.model.conv2d_2a.conv.padding,
                                                        dilation=self.model.conv2d_2a.conv.dilation,
                                                        groups=self.model.conv2d_2a.conv.groups,
                                                        bias=self.model.conv2d_2a.conv.bias is not None))
        setattr(self.model.conv2d_2a, 'bn', nn.BatchNorm2d(num_features=self.start_width,
                                                           eps=self.model.conv2d_2a.bn.eps,
                                                           momentum=self.model.conv2d_2a.bn.momentum,
                                                           affine=self.model.conv2d_2a.bn.affine,
                                                           track_running_stats=self.model.conv2d_2a.bn.track_running_stats))
        setattr(self.model.conv2d_2b, 'conv', nn.Conv2d(in_channels=self.start_width,
                                                        out_channels=self.model.conv2d_2b.conv.out_channels,
                                                        kernel_size=self.model.conv2d_2b.conv.kernel_size,
                                                        stride=self.model.conv2d_2b.conv.stride,
                                                        padding=self.model.conv2d_2b.conv.padding,
                                                        dilation=self.model.conv2d_2b.conv.dilation,
                                                        groups=self.model.conv2d_2b.conv.groups,
                                                        bias=self.model.conv2d_2b.conv.bias is not None))
        setattr(self.model.conv2d_2b, 'bn', nn.BatchNorm2d(num_features=self.model.conv2d_2b.conv.out_channels,
                                                           eps=self.model.conv2d_2b.bn.eps,
                                                           momentum=self.model.conv2d_2b.bn.momentum,
                                                           affine=self.model.conv2d_2b.bn.affine,
                                                           track_running_stats=self.model.conv2d_2b.bn.track_running_stats))

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x) if self.sigmoid_on else x
        return x

    def to(self, device):
        self.model.to(device)
        self.sigmoid.to(device)


class AdaptedInputXception(nn.Module):
    def __init__(self, input_size, start_width=32, n_classes=1, sigmoid=True, num_classes=1000, pretrained=False):
        super().__init__()
        self.input_size = input_size
        self.planes = self.input_size[0]
        self.start_width = start_width
        self.name = f'Xception with {self.planes}-Planed Stem'
        self.sigmoid_on = sigmoid
        self.sigmoid = nn.Sigmoid()

        self.model = xception(num_classes=num_classes, pretrained=pretrained)
        setattr(self.model, 'conv1', nn.Conv2d(in_channels=self.planes,
                                               out_channels=self.start_width,
                                               kernel_size=self.model.conv1.kernel_size,
                                               stride=self.model.conv1.stride,
                                               padding=self.model.conv1.padding,
                                               dilation=self.model.conv1.dilation,
                                               groups=self.model.conv1.groups,
                                               bias=self.model.conv1.bias is not None))
        setattr(self.model, 'bn1', nn.BatchNorm2d(num_features=self.start_width,
                                                  eps=self.model.bn1.eps,
                                                  momentum=self.model.bn1.momentum,
                                                  affine=self.model.bn1.affine,
                                                  track_running_stats=self.model.bn1.track_running_stats))
        setattr(self.model, 'conv2', nn.Conv2d(in_channels=self.start_width,
                                               out_channels=self.model.conv2.out_channels,
                                               kernel_size=self.model.conv2.kernel_size,
                                               stride=self.model.conv2.stride,
                                               padding=self.model.conv2.padding,
                                               dilation=self.model.conv2.dilation,
                                               groups=self.model.conv2.groups,
                                               bias=self.model.conv2.bias is not None))
        setattr(self.model, 'bn2', nn.BatchNorm2d(num_features=self.model.conv2.out_channels,
                                                  eps=self.model.bn2.eps,
                                                  momentum=self.model.bn2.momentum,
                                                  affine=self.model.bn2.affine,
                                                  track_running_stats=self.model.bn2.track_running_stats))

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x) if self.sigmoid_on else x
        return x

    def to(self, device):
        self.model.to(device)
        self.sigmoid.to(device)


def update_layer_channels(model, new_channels, old_channels):
    updated_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            updated_layers[name] = nn.Conv2d(
                in_channels=int(np.round(new_channels * module.in_channels / old_channels)),
                out_channels=int(np.round(new_channels * module.out_channels / old_channels)),
                kernel_size=module.kernel_size, stride=module.stride, padding=module.padding,
                dilation=module.dilation, groups=module.groups, bias=module.bias is not None)
        elif isinstance(module, nn.BatchNorm2d):
            updated_layers[name] = nn.BatchNorm2d(
                num_features=int(np.round(new_channels * module.num_features / old_channels)),
                eps=module.eps, momentum=module.momentum, affine=module.affine,
                track_running_stats=module.track_running_stats)
    for name, layer in updated_layers.items():
        parts = name.split('.')
        current_module = model
        for part in parts[:-1]:
            current_module = getattr(current_module, part)
        setattr(current_module, parts[-1], layer)

# endregion
