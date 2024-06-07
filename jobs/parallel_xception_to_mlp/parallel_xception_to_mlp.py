import os

import torch
from torch import nn
from pretrainedmodels import xception
import torch.multiprocessing as mp

from my_modules.model_learning.loader_maker import split_augmented_data
from my_modules.model_learning.model_metrics import score_model
from my_modules.nsclc import NSCLCDataset
from my_modules.custom_models import MLPNet as MLP, FeatureExtractorToClassifier as FETC


def main():
    # Set up multiprocessing context
    mp.set_start_method('forkserver', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create data with psuedo-RGB stack for each mode
    data = NSCLCDataset('data/NSCLC_Data_for_ML',
                        mode=['orr', 'taumean', 'boundfraction'], label='M', mask_on=False, device='cpu')
    data.augment()
    data.transform_to_psuedo_rgb()
    data.normalize_channels_to_max()
    data.to(device)

    # Prep output dirs and files
    os.makedirs('outputs/plots', exist_ok=True)
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
    classifier = MLP

    # Set up hyperparameters
    batch_size = 64
    learning_rates = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
    epochs = [5, 10, 50, 125, 250, 500, 1000, 2500]
    optim_fn = torch.optim.Adam
    loss_fn = nn.BCEWithLogitsLoss()

    # Prepare data loaders
    train_set, eval_set, test_set = split_augmented_data(data, augmentation_factor=5, split=(0.75, 0.15, 0.1))
    training_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                                  drop_last=(True if len(train_set) % batch_size == 1 else False))
    evaluation_loader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size, shuffle=False, num_workers=0,
                                                    drop_last=(True if len(eval_set) % batch_size == 1 else False))
    testing_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,
                                                 drop_last=(True if len(test_set) % batch_size == 1 else False))

    # Iterate LR
    individual_training_loss = []
    ensemble_training_loss = []
    parallel_training_loss = []
    individual_eval_losses = []
    ensemble_eval_loss = []
    parallel_eval_loss = []
    individual_eval_scores = []
    ensemble_eval_scores = []
    parallel_eval_scores = []
    individual_test_scores = []
    ensemble_test_scores = []
    parallel_test_scores = []
    for lr in learning_rates:
        # Create models
        # Individual and ensemble-averaging models
        models = [FETC(data.shape[1:], feature_extractor=feature_extractor, classifier=classifier, layer='conv4')
                  for _ in range(data.stack_height)]
        [model.to(device) for model in models]
        ensemble_combiner = nn.Linear(data.stack_height, 1).to(device)  # Learnable linear combination of logits

        # Parallel feature extraction to single net model with input size for all models features
        input_size = sum([model.feature_map_dims[1] for model in models])
        fetc_parallel_classifier = MLP(input_size).to(device)

        # Set up optimizers
        optimizers = [optim_fn(model.parameters(), lr=lr) for model in models]
        ensemble_optimizer = optim_fn(ensemble_combiner.parameters(), lr=lr)
        parallel_optimizer = optim_fn(fetc_parallel_classifier.parameters(), lr=lr)

        # Iterate epochs
        epoch = sorted(epochs)[-1]
        individual_training_loss.append(data.stack_height * [[]])
        ensemble_training_loss.append([])
        parallel_training_loss.append([])

        individual_eval_losses.append(data.stack_height * [[]])
        ensemble_eval_loss.append([])
        parallel_eval_loss.append([])

        individual_eval_scores.append(data.stack_height * [[]])
        ensemble_eval_scores.append([])
        parallel_eval_scores.append([])

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
                [optimizer.zero_grad() for optimizer in optimizers]
                outs = [model(x[:, ch].squeeze(1)) for ch, model in enumerate(models)]
                losses = [loss_fn(out, target.unsqueeze(1)) for out in outs]
                [loss.backward() for loss in losses]
                for individual_loss, loss in zip(individual_losses, losses):
                    individual_loss += loss.item()
                [optimizer.step() for optimizer in optimizers]

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
                individual_training_loss.append(il / len(train_set))
            ensemble_training_loss[-1].append(ensemble_loss / len(train_set))
            parallel_training_loss[-1].append(parallel_loss / len(train_set))

            # Validation
            individual_losses = data.stack_height * [0]
            ensemble_loss = 0
            parallel_loss = 0
            # Iterate through dataloader
            with torch.no_grad():
                psuedo_loaders = data.stack_height * [[]]  # For scoring
                individual_output_loader = []
                feature_loader = []
                for x, target in evaluation_loader:
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
                    [optimizer.zero_grad() for optimizer in optimizers]
                    outs = []
                    for ch, (model, pl) in enumerate(zip(models, psuedo_loaders)):
                        outs.append(model(x[:, ch].squeeze(1)))
                        # Make a psuedo-loader for each model to use for scoring
                        pl.append((x[:, ch].squeeze(1), target))

                    losses = [loss_fn(out, target.unsqueeze(1)) for out in outs]
                    for individual_loss, loss in zip(individual_losses, losses):
                        individual_loss += loss.item()

                    # Make output loader for ensemble model
                    individual_output_loader.append((torch.cat(outs, dim=1).detach(), target))

                    # Determine ensemble out
                    ensemble_optimizer.zero_grad()
                    out = ensemble_combiner(torch.cat(outs, dim=1).detach())
                    loss = loss_fn(out, target.unsqueeze(1))
                    ensemble_loss += loss.item()

                    # Feed parallel-extracted features into full classifier
                    parallel_optimizer.zero_grad()
                    out = fetc_parallel_classifier(torch.stack(features, dim=1).detach())
                    loss = loss_fn(out, target.unsqueeze(1))
                    parallel_loss += loss.item()

            # Update validation losses
            for il, iel in zip(individual_losses, individual_eval_losses[-1]):
                iel.append(il / len(eval_set))
            ensemble_eval_loss[-1].append(ensemble_loss / len(eval_set))
            parallel_eval_loss[-1].append(parallel_loss / len(eval_set))

            # Update validation scores
            for score, model, loader in zip(individual_eval_scores[-1], models, psuedo_loaders):
                score.append(score_model(model, loader))
            ensemble_eval_scores[-1].append(score_model(ensemble_combiner, individual_output_loader))
            parallel_eval_scores[-1].append(score_model(fetc_parallel_classifier, feature_loader))

            # Write outputs
            with open('outputs/individual_results.txt', 'a') as f:
                f.write(f'LR: {lr} - Epoch: {ep + 1}')
                for mode, score in zip(data.mode, individual_test_scores[-1]):
                    f.write(f'\n{mode}')
                    for key, item in score.items():
                        if 'Confusion' not in key:
                            f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|\n')

            with open('outputs/ensemble_results.txt', 'a') as f:
                f.write(f'LR: {lr} - Epoch: {ep + 1}\n')
                for key, item in ensemble_eval_scores[-1].items():
                    if 'Confusion' not in key:
                        f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|\n')

            with open('outputs/parallel_results.txt', 'a') as f:
                f.write(f'LR: {lr} - Epoch: {ep + 1}\n')
                for key, item in parallel_eval_scores[-1].items():
                    if 'Confusion' not in key:
                        f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|\n')

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
                        [optimizer.zero_grad() for optimizer in optimizers]
                        outs = []
                        for ch, (model, pl) in enumerate(zip(models, psuedo_loaders)):
                            outs.append(model(x[:, ch].squeeze(1)))
                            # Make a psuedo-loader for each model to use for scoring
                            pl.append((x[:, ch].squeeze(1), target))

                        # Make output loader for ensemble model
                        individual_output_loader.append((torch.cat(outs, dim=1).detach(), target))

                # Update testing scores
                for score, model, loader in zip(individual_test_scores[-1], models, psuedo_loaders):
                    score.append(score_model(model, loader))
                ensemble_test_scores[-1].append(score_model(ensemble_combiner, individual_output_loader))
                parallel_test_scores[-1].append(score_model(fetc_parallel_classifier, feature_loader))

                # Write outputs
                with open('outputs/individual_results.txt', 'a') as f:
                    f.write(f'\n>>>LR: {lr} - Epoch: {ep + 1}')
                    for mode, score in zip(data.mode, individual_test_scores[-1]):
                        f.write(f'\n{mode}______________________________________________\n')
                        for key, item in score.items():
                            if 'Confusion' not in key:
                                f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|\n')
                        f.write('_____________________________________________________\n')
                    f.write(f'<<<\n')

                with open('outputs/ensemble_results.txt', 'a') as f:
                    f.write(f'\n>>>LR: {lr} - Epoch: {ep + 1}')
                    f.write('_____________________________________________________\n')
                    for key, item in ensemble_test_scores[-1].items():
                        if 'Confusion' not in key:
                            f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|\n')
                    f.write('_____________________________________________________\n')

                with open('outputs/parallel_results.txt', 'a') as f:
                    f.write(f'\n>>>LR: {lr} - Epoch: {ep + 1}')
                    f.write('_____________________________________________________\n')
                    for key, item in parallel_test_scores[-1].items():
                        if 'Confusion' not in key:
                            f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|\n')
                    f.write('_____________________________________________________\n')


if __name__ == '__main__':
    main()
