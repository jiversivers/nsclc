import torch.optim as optim

from my_modules.custom_models import (MLPNet, ParallelMLPNet, RegularizedMLPNet, RegularizedParallelMLPNet,
                                      CNNet, ParallelCNNet, RegularizedCNNet, RegularizedParallelCNNet)
from my_modules.model_learning import single_model_iterator
from my_modules.nsclc.nsclc_dataset import NSCLCDataset
from torch import nn


def main():
    # Subset name
    subname = 'image_augmented_response'

    # Prepare data
    data = NSCLCDataset('data/NSCLC_Data_for_ML', ['orr', 'photons', 'taumean', 'boundfraction'], label='Response')
    data.normalize_channels_to_max()
    data.augment()

    # Prepare training/data-loading parameters
    optim_fun = optim.SGD
    criterion = nn.BCELoss()
    bs = 64  # Batch size

    # Iterable hyperparameters
    hyperparameters = {'lightning': {'LR': 0.1, 'EP': 100},
                       'fast': {'LR': 0.01, 'EP': 150},
                       'mid': {'LR': 0.001, 'EP': 500},
                       'slow': {'LR': 0.00001, 'EP': 2500}}

    # Init results
    results = {}
    results_file_path = f'results_{subname}.txt'
    with open(results_file_path, 'w') as results_file:
        results_file.write(f'Results for {subname}\n')

    # Iterate hist-based classifiers (RNNs, no CNNs)
    models = [MLPNet, CNNet, ParallelMLPNet, ParallelCNNet,
              RegularizedMLPNet, RegularizedCNNet, RegularizedParallelMLPNet, RegularizedParallelCNNet]
    for style, hp in hyperparameters.items():
        print(f'Currently training at {style} rate...')
        results[style] = single_model_iterator(models, {subname: data},
                                               hp['EP'], bs, criterion,
                                               optim_fun, lr=hp['LR'],
                                               num_workers=(24, 6, 2),
                                               pin_memory=True, momentum=0.9)
        with open(results_file_path, 'a') as f:
            for mod, res in results[style][subname].items():
                f.write(f'{subname} at {style} rate -- {mod} Accuracy: {res}\n')

        print(results)


if __name__ == "__main__":
    main()