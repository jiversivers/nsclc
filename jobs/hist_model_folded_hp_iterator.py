import os

import torch.nn

from my_modules.custom_models import MLPNet, RegularizedParallelMLPNet, RegularizedParallelRNNet
from my_modules.model_learning import fold_cross_validate, masked_loss
from my_modules.model_learning.loader_maker import fold_augmented_data
from my_modules.nsclc import NSCLCDataset


def main():
    # Set up dataset
    data = NSCLCDataset('data/NSCLC_Data_for_ML', ['orr', 'taumean', 'boundfraction'], device='cpu',
                        label='Metastases', mask_on=True)
    data.augment()
    data.normalize_channels_to_max()
    data.dist_transformed()

    # Fold dataset
    data_folds = fold_augmented_data(data, augmentation_factor=5, num_folds=5)

    # Set up hyperparameters
    bs = 64
    learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    epochs = [125, 250, 500, 100, 2500]
    loss_fn = masked_loss(torch.nn.BCELoss(reduction='none'))
    optimizer = [torch.optim.SGD, {'momentum': 0.9}]

    # Set up models to try
    model_fns = [MLPNet, RegularizedParallelMLPNet, RegularizedParallelRNNet]

    # Make model save dir
    os.makedirs('models', exist_ok=True)

    for lr in learning_rates:
        for ep in epochs:
            for model_fn in model_fns:
                accuracy, running_loss, models = fold_cross_validate(model_fn, data_folds,
                                                                     learning_rate=lr, epochs=ep, batch_size=bs,
                                                                     loss_fn=loss_fn, masked_loss_fn=True,
                                                                     optimizer_fn=optimizer[0], **optimizer[1])
                for fold, model in enumerate(models):
                    os.makedirs(f'models/{model.name}', exist_ok=True)
                    torch.save(model.state_dict(), f'models/{model.name}/{model.name}_{lr}_{ep}_Fold{fold + 1}.pt')

                # Hard clear cache to try to keep memory clear
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
