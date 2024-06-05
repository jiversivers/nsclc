import os

import torch
import torch.multiprocessing as mp
from matplotlib import pyplot as plt
from torch import nn

from my_modules.custom_models import MLPNet, RegularizedParallelMLPNet, RegularizedMLPNet, \
    ParallelMLPNet, CNNet, RegularizedCNNet, ParallelCNNet, RegularizedParallelCNNet
from my_modules.model_learning.loader_maker import split_augmented_data
from my_modules.model_learning.model_metrics import calculate_auc_roc, score_model
from my_modules.nsclc import NSCLCDataset


def main():
    # Set up multiprocessing context
    mp.set_start_method('forkserver', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create data with psuedo-RGB stack for each mode
    data = NSCLCDataset(
        'data/NSCLC_Data_for_ML',
        mode=['orr', 'taumean', 'boundfraction'], label='M', mask_on=False, device='cpu')
    data.augment()
    data.normalize_channels_to_max()
    data.to(device)

    # Set up hyperparameters
    batch_size = 64
    learning_rates = [0.001, 0.0001, 0.00001, 0.000001]
    epochs = [5, 10, 50, 125, 250, 500, 1000]
    epoch = sorted(epochs)[-1]
    optim_fn = torch.optim.RMSprop
    loss_fn = nn.TripletMarginLoss()

    # Split dataset
    training_set, validation_set, testing_set = split_augmented_data(data, augmentation_factor=5, split=(.75, .15, .1))
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                                  drop_last=(True if len(training_set) % batch_size == 1 else False))
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=0,
                                                    drop_last=(
                                                        True if len(validation_set) % batch_size == 1 else False))
    testing_loader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=False, num_workers=0,
                                                 drop_last=(True if len(testing_set) % batch_size == 1 else False))

    # Set up models to try
    model_fns = [MLPNet, RegularizedMLPNet, ParallelMLPNet, RegularizedParallelMLPNet,
                 CNNet, RegularizedCNNet, ParallelCNNet, RegularizedParallelCNNet]

    # Make model save dir
    os.makedirs('outputs/', exist_ok=True)
    with open('outputs/overview_results.txt', 'w') as f:
        f.write('Overview of Model Performance\n')

    for model_fn in model_fns:
        os.makedirs(f'outputs/{model_fn.__name__}/plots', exist_ok=True)
        os.makedirs(f'outputs/{model_fn.__name__}/prints', exist_ok=True)
        for lr in learning_rates:
            # Prep results print file
            with open(f'outputs/{model_fn.__name__}/prints/lr{lr}_results.txt', 'w') as f:
                f.write(f'Results\n')
            # Make fresh model and optimizer
            model = model_fn(data.shape)
            model.to(device)
            optimizer = optim_fn(model.parameters(), lr=lr)

            # Evaluation metrics
            running_auc = []
            running_accuracy = []
            running_figs = []

            # Training
            model.train()
            for ep in range(epoch):
                # Model the entire loader
                outs = torch.tensor([]).to(device)
                targets = torch.tensor([]).to(device)
                for x, target in training_loader:
                    outs = torch.cat((outs, model(x)), dim=0)
                    targets = torch.cat((targets, target), dim=0)

                # Regroup outputs based on class (random by first instance
                anchor = outs[targets == target[0]]
                negative = outs[targets != targets[0]]

                # Truncate to the smallest class
                truncation_index = min(anchor.shape[0], negative.shape[0])
                # Make positives from any leftovers with reordered data to prevent self-overlap with anchor
                positive = torch.cat((anchor[truncation_index:], anchor[0:truncation_index]), dim=0)
                positive = positive[:truncation_index]
                anchor = anchor[:truncation_index]
                negative = negative[:truncation_index]

                # Calculate loader-wide loss, backprop, and learn
                loss = loss_fn(anchor, positive, negative)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Validation
                model.eval()
                validation_loss = 0.0
                with torch.no_grad():
                    targets = torch.tensor([]).to(device)
                    for x, target in validation_loader:
                        out = model(x)
                        targets = torch.cat((targets, target), dim=0)
                        validation_loss += nn.BCEWithLogitsLoss()(out, target.unsqueeze(1))
                    scores = score_model(model, validation_loader)
                    with open(f'outputs/{model_fn.__name__}/prints/lr{lr}_results.txt', 'a') as f:
                        for key, item in scores.items():
                            if 'Confusion' not in key:
                                f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|\n')

                # Testing
                if ep + 1 in epochs:
                    scores, fig = score_model(model, testing_loader, make_plot=True, threshold_type='ROC')
                    with open(f'outputs/{model_fn.__name__}/prints/lr{lr}_results.txt', 'a') as f:
                        f.write('\n_____________________________________________________\n')
                        for key, item in scores.items():
                            if 'Confusion' not in key:
                                f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|\n')
                        f.write('\n_____________________________________________________\n')
                    running_auc.append(scores['ROC-AUC'])
                    running_accuracy.append(scores['Balanced Accuracy at Threshold'])
                    running_figs.append(fig)

            # Determine the best performance first by auc, then acc if there are ties
            best_auc = max(running_auc)
            max_auc_loc = [a == best_auc for a in running_auc]  # Find where the max auc(s) is/are
            accs_for_best_auc = [running_accuracy[idx] for idx, mal in enumerate(max_auc_loc) if
                                 mal]  # Find the AUC at those
            running_figs[running_accuracy.index(max(accs_for_best_auc))].savefig(  # Save fig from that time
                f'outputs/{model_fn.__name__}/plots/best_auc_acc{lr}.png')
            for fig in running_figs:
                plt.close(fig)
            best_acc = max(accs_for_best_auc)
            best_epoch = epochs[running_auc.index(best_auc)]

            # Plot training metrics over testing ranges
            fig, ax1 = plt.subplots()
            ax1.plot(epochs, running_auc, 'r-', label='AUC')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('AUC')
            ax1.set_ylim([0, 1])
            ax1.tick_params(axis='both', labelcolor='r')

            ax2 = ax1.twinx()
            ax2.plot(epochs, running_accuracy, 'b-', label='Best Balanced Accuracy')
            ax2.set_ylabel('Accuracy')
            ax2.set_ylim([0, 1])
            ax2.tick_params(axis='y', labelcolor='b')

            fig.suptitle(f'Testing performance over training at learning rate of {lr}')
            ax1.legend(loc='upper left')
            ax2.legend(loc='lower right')

            fig.savefig(f'outputs/{model_fn.__name__}/plots/test_metrics_lr{lr}.png')
            plt.close(fig)

            with open('outputs/overview_results.txt', 'a') as f:
                f.write(f'\n{model.name} Best balanced accuracy of {100 * best_acc:.2f}% and best AUC of {best_auc:.2f}'
                        f' achieved at epoch {best_epoch} with learning rate of {lr}')

            # Hard clear cache to try to keep memory clear on GPU
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
