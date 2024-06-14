import os

import torch
import torch.multiprocessing as mp
import torchvision.transforms.v2 as tvt
from matplotlib import pyplot as plt
from pretrainedmodels import xception
from torch import nn

from my_modules.custom_models import CometClassifierWithBinaryOutput as Comet, FeatureExtractorToClassifier as FETC
from my_modules.model_learning.model_metrics import score_model
from my_modules.nsclc import NSCLCDataset
from my_modules.nsclc import patient_wise_train_test_splitter

def main():
    # Set up multiprocessing context
    mp.set_start_method('forkserver', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create data with psuedo-RGB stack for each mode
    data = NSCLCDataset('data/NSCLC_Data_for_ML',
                        mode=['orr', 'taumean', 'boundfraction'], label='M', mask_on=False, device='cpu')
    data.augment()
    data.transform_to_psuedo_rgb()
    data.normalize_channels('preset')
    data.transforms = tvt.Compose([tvt.RandomVerticalFlip(p=0.25),
                                   tvt.RandomHorizontalFlip(p=0.25),
                                   tvt.RandomRotation(degrees=(-180, 180))])

    # Prep output dirs and files
    os.makedirs('outputs/plots/individual', exist_ok=True)
    os.makedirs('outputs/plots/ensemble', exist_ok=True)
    os.makedirs('outputs/plots/parallel', exist_ok=True)
    # For individual mode models
    with open('outputs/individual_results.txt', 'w') as f:
        f.write(f'Individual Results\n'
                f'{data.mode}')
    # For ensemble
    with open('outputs/ensemble_results.txt', 'w') as f:
        f.write(f'Ensemble Results')
    # For parallel
    with open('outputs/parallel_results.txt', 'w') as f:
        f.write(f'Parallel Results')

    # Define our base feature extractor and turn the gradients off -- we won't train it, just use it to feed our MLP.
    feature_extractor = xception(num_classes=1000, pretrained=False)

    # Load pretrained from download
    state_dict = torch.load(r'/home/jdivers/data/torch_checkpoints/pretrained_models/xception-43020ad28.pth')
    state_dict['last_linear.weight'] = state_dict.pop('fc.weight')
    state_dict['last_linear.bias'] = state_dict.pop('fc.bias')

    feature_extractor.load_state_dict(state_dict)
    for params in feature_extractor.parameters():
        params.requires_grad = False

    # Define classifier
    classifier = Comet

    # Set up hyperparameters
    batch_size = 64
    learning_rates = [5e-5, 5e-6, 1e-6, 5e-7]
    epochs = [125, 250, 500, 1000, 2000]
    optim_fn = torch.optim.Adam
    loss_fn = nn.BCEWithLogitsLoss()

    # Prepare data loaders
    train_set, test_set = patient_wise_train_test_splitter(data, n=3)
    training_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                                  drop_last=(True if len(train_set) % batch_size == 1 else False))
    testing_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,
                                                 drop_last=(True if len(test_set) % batch_size == 1 else False))

    # Iterate LR
    individual_training_loss = []
    ensemble_training_loss = []
    parallel_training_loss = []
    individual_test_scores = []
    ensemble_test_scores = []
    parallel_test_scores = []
    for lr in learning_rates:
        # Create models
        # Individual and ensemble-averaging models
        models = [
            FETC(data.shape[1:], feature_extractor=feature_extractor, classifier=classifier, layer='conv2d_7b')
            for _ in range(data.stack_height)]
        [model.to(device) for model in models]
        ensemble_combiner = nn.Linear(data.stack_height, 1).to(device)  # Learnable linear combination of logits

        # Parallel feature extraction to single net model with input size for all models features
        input_size = sum([model.feature_map_dims[1] for model in models])
        fetc_parallel_classifier = classifier(input_size).to(device)

        # Set up optimizers
        individual_optimizers = [optim_fn(model.classifier.parameters(), lr=lr) for model in models]
        ensemble_optimizer = optim_fn(ensemble_combiner.parameters(), lr=lr)
        parallel_optimizer = optim_fn(fetc_parallel_classifier.parameters(), lr=lr)

        # Iterate epochs
        epoch = sorted(epochs)[-1]
        individual_training_loss.append(data.stack_height * [[]])
        ensemble_training_loss.append([])
        parallel_training_loss.append([])

        individual_test_scores.append(data.stack_height * [[]])
        ensemble_test_scores.append([])
        parallel_test_scores.append([])

        for ep in range(epoch):
            # Training
            individual_losses = data.stack_height * [0]
            ensemble_loss = 0
            parallel_loss = 0
            # Iterate through dataloader
            for x, target in training_loader:
                # Mode-wise models (for individual and ensemble architectures)
                with torch.no_grad():
                    # Get feature maps, avg, and flatten (just like in the whole model)
                    features = [model.flat(model.global_avg_pool(model.get_features(x[:, ch].squeeze(1))))
                                for ch, model in enumerate(models)]

                # Get final output for each model and do backprop
                outs = []
                for ch, (optimizer, model) in enumerate(zip(individual_optimizers, models)):
                    optimizer.zero_grad()
                    out = model(x[:, ch].squeeze(1))
                    outs.append(out)
                    loss = loss_fn(out, target.unsqueeze(1))
                    individual_losses[ch] += loss.item()
                    loss.backward()
                    optimizer.step()

                # Get ensemble output and do backprop
                ensemble_optimizer.zero_grad()
                out = ensemble_combiner(torch.cat(outs, dim=1).detach())
                loss = loss_fn(out, target.unsqueeze(1))
                loss.backward()
                ensemble_loss += loss.item()
                ensemble_optimizer.step()

                # Feed parallel-extracted features into full classifier
                parallel_optimizer.zero_grad()
                out = fetc_parallel_classifier(torch.stack(features, dim=1).detach())
                loss = loss_fn(out, target.unsqueeze(1))
                loss.backward()
                parallel_loss += loss.item()
                parallel_optimizer.step()

            # Update training checks
            for il, itl in zip(individual_losses, individual_training_loss[-1]):
                itl.append(il / len(train_set))
            ensemble_training_loss[-1].append(ensemble_loss / len(train_set))
            parallel_training_loss[-1].append(parallel_loss / len(train_set))
            print(f'Epoch: {ep + 1} | Individual Loss: {individual_training_loss[-1]}'
                  f' | Ensemble Loss: {ensemble_training_loss[-1]}'
                  f' | Parallel Loss: {parallel_training_loss[-1]}')
            # If we are at a checkpoint epoch
            if ep + 1 in epochs:
                # Testing
                # Iterate through dataloader
                with torch.no_grad():
                    psuedo_loaders = data.stack_height * [[]]  # For scoring
                    individual_output_loader = []
                    feature_loader = []
                    for x, target in testing_loader:
                        # Put onto GPU if not
                        if torch.cuda.is_available():
                            x = x.cuda() if x.device.type != 'cuda' else x
                            target = target.cuda() if target.device.type != 'cuda' else target
                        # Mode-wise models (for individual and ensemble architectures)
                        # Get feature maps, avg, and flatten (just like in the whole model)
                        features = [model.flat(model.global_avg_pool(model.get_features(x[:, ch].squeeze(1))))
                                    for ch, model in enumerate(models)]
                        feature_loader.append((torch.cat(features, dim=1).detach(), target))

                        # Get final output for each model
                        outs = []
                        for ch, (model, loader) in enumerate(zip(models, psuedo_loaders)):
                            out = model(x[:, ch].squeeze(1))
                            outs.append(out)
                            # Make a psuedo-loader for each individual model to use for scoring
                            loader.append(x[:, ch].unsqueeze(1), target)

                        # Make output loader for ensemble model
                        individual_output_loader.append((torch.cat(outs, dim=1).detach(), target))

                # Update testing scores
                for score, model, loader in zip(individual_test_scores[-1], models, psuedo_loaders):
                    auc_score, fig = score_model(model, loader, make_plot=True, threshold_type='roc')
                    score.append(auc_score)
                    fig.savefig(f'outputs/plots/individual/auc_{model.name}_{ep}_{lr}.png')
                    plt.close(fig)
                auc_score, fig = score_model(ensemble_combiner, individual_output_loader,
                                             make_plot=True, threshold_type='roc')
                ensemble_test_scores[-1].append(auc_score)
                fig.savefig(f'outputs/plots/ensemble/auc_ensemble_{ep}_{lr}.png')
                plt.close(fig)
                auc_score, fig = score_model(fetc_parallel_classifier, feature_loader,
                                             make_plot=True, threshold_type='roc')
                parallel_test_scores[-1].append(auc_score)
                fig.savefig(f'outputs/plots/parallel/auc_parallel_{ep}_{lr}.png')
                plt.close(fig)

                # Write outputs
                with open('outputs/individual_results.txt', 'a') as f:
                    f.write(f'\n>>>LR: {lr} - Epoch: {ep + 1}')
                    for mode, score in zip(data.mode, individual_test_scores[-1]):
                        f.write(f'\n{mode}______________________________________________\n')
                        for key, item in score[-1].items():
                            if 'Confusion' not in key:
                                f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|\n')
                        f.write('_____________________________________________________\n')
                    f.write(f'<<<\n')

                with open('outputs/ensemble_results.txt', 'a') as f:
                    f.write(f'\n>>>LR: {lr} - Epoch: {ep + 1}')
                    f.write('_____________________________________________________\n')
                    for key, item in ensemble_test_scores[-1][-1].items():
                        if 'Confusion' not in key:
                            f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|\n')
                    f.write('_____________________________________________________\n')

                with open('outputs/parallel_results.txt', 'a') as f:
                    f.write(f'\n>>>LR: {lr} - Epoch: {ep + 1}')
                    f.write('_____________________________________________________\n')
                    for key, item in parallel_test_scores[-1][-1].items():
                        if 'Confusion' not in key:
                            f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|\n')
                    f.write('_____________________________________________________\n')

if __name__ == '__main__':
    main()
