import os

import torch
from torch import nn
from pretrainedmodels import xception
import torch.multiprocessing as mp

from my_modules.model_learning.loader_maker import split_augmented_data
from my_modules.nsclc import NSCLCDataset
from my_modules.custom_models import RegularizedMLPNet as RegMLP, FeatureExtractorToClassifier as FETC


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

    # Prep output dirs and files
    os.makedirs('outputs/plots', exist_ok=True)
    # For individual mode models
    with open('outputs/individual_results.txt', 'w') as results_file:
        results_file.write(f'Individual Results\n'
                           f'{data.mode}')
    # For ensemble
    with open('outputs/ensemble_results.txt', 'w') as results_file:
        results_file.write(f'Ensemble Results')
    # For parallel
    with open('outputs/parallel_results.txt', 'w') as results_file:
        results_file.write(f'Parallel Results')

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
    classifier = RegMLP

    # Set up hyperparameters
    batch_size = 64
    learning_rates = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
    epochs = [5, 10, 50, 125, 250, 500, 1000, 2500]
    optim_fn = torch.optim.Adam
    loss_fn = nn.BCELoss()

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
    parallel_training_loss = []
    individual_eval_loss = []
    parallel_eval_loss = []
    individual_eval_accuracies = []
    ensemble_eval_accuracy = []
    parallel_eval_accuracy = []
    individual_test_accuracies = []
    ensemble_test_accuracy = []
    parallel_test_accuracy = []
    for lr in learning_rates:
        # Create models
        # Individual and ensemble-averaging models
        models = [FETC(data.shape[1:], feature_extractor=feature_extractor, classifier=classifier, layer='conv4')
                  for _ in range(len(data.mode))]
        [model.to(device) for model in models]

        # Parallel feature extraction to single net model with input size for all models features
        input_size = sum([model.feature_map_dims[1] for model in models])
        fetc_parallel_classifier = RegMLP(input_size)

        # Set up optimizers
        optimizers = [optim_fn(model.parameters(), lr=lr) for model in models]
        parallel_optimizer = optim_fn(fetc_parallel_classifier.parameters(), lr=lr)

        # Iterate epochs
        epoch = sorted(epochs)[-1]
        individual_training_loss.append([])
        parallel_training_loss.append([])

        individual_eval_loss.append([])
        parallel_eval_loss.append([])

        individual_eval_accuracies.append([])
        ensemble_eval_accuracy.append([])
        parallel_eval_accuracy.append([])

        individual_test_accuracies.append([])
        ensemble_test_accuracy.append([])
        parallel_test_accuracy.append([])
        for ep in range(epoch):
            # Training
            individual_losses = len(data.mode) * [0]
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
                losses = [loss_fn(out, target.unsqueeze()) for out in outs]
                [loss.backward() for loss in losses]
                for individual_loss, loss in zip(individual_losses, losses):
                    individual_loss += loss.item()
                [optimizer.step() for optimizer in optimizers]

                # Feed parallel-extracted features into full classifier
                parallel_optimizer.zero_grad()
                out = fetc_parallel_classifier(torch.stack(features, dim=1))
                loss = loss_fn(out, target.unsqueeze())
                loss.backward()
                parallel_loss += loss.item()
                parallel_optimizer.step()

            # Update training checks
            individual_training_loss[-1] = [individual_loss / len(train_set) for individual_loss in individual_losses]
            parallel_training_loss[-1] = parallel_loss / len(train_set)

            # Validation
            individual_losses = len(data.mode) * [0]
            parallel_loss = 0
            individual_corrects = len(data.mode) * [0]
            ensemble_correct = 0
            parallel_correct = 0
            # Iterate through dataloader
            with torch.no_grad():
                for x, target in evaluation_loader:
                    # Mode-wise models (for individual and ensemble architectures)
                    # Get feature maps, avg, and flatten (just like in the whole model)
                    features = [model.flat(model.global_avg_pool(model.get_features(x[:, ch].squeeze(1))))
                                for ch, model in enumerate(models)]

                    # Get final output for each model and do backprop
                    [optimizer.zero_grad() for optimizer in optimizers]
                    outs = [model(x[:, ch].squeeze(1)) for ch, model in enumerate(models)]
                    preds = [torch.round(out) for out in outs]
                    individual_corrects = [torch.sum(pred == target.unsqueeze(1)) for pred in preds]
                    losses = [loss_fn(out, target.unsqueeze()) for out in outs]
                    for individual_loss, loss in zip(individual_losses, losses):
                        individual_loss += loss.item()

                    # Determine ensemble prediction and accuracy
                    pred = torch.round(torch.mean(torch.stack(outs, dim=0), dim=0))
                    ensemble_correct += torch.sum(pred == target.unsqueeze(1))

                    # Feed parallel-extracted features into full classifier
                    parallel_optimizer.zero_grad()
                    out = fetc_parallel_classifier(torch.stack(features, dim=1))
                    pred = torch.round(out)
                    parallel_correct = torch.sum(pred == target.unsqueeze(1))
                    loss = loss_fn(out, target.unsqueeze())
                    parallel_loss += loss.item()

            # Update validation checks
            individual_eval_loss[-1] = [individual_loss / len(eval_set) for individual_loss in individual_losses]
            parallel_eval_loss[-1] = parallel_loss / len(eval_set)

            individual_eval_accuracies[-1] = [correct / len(eval_set) for correct in individual_corrects]
            ensemble_eval_accuracy[-1] = ensemble_correct / len(eval_set)
            parallel_eval_accuracy[-1] = parallel_correct / len(eval_set)

            # Write outputs
            with open('outputs/individual_results.txt', 'a') as results_file:
                results_file.write(f'\nEpoch: {ep}:')
                for mode, loss, accu in zip(data.mode, individual_training_loss[-1], individual_eval_accuracies[-1]):
                    results_file.write(f'-- {mode}: Loss-{loss:.4f}. Accu-{accu} -- ')

            with open('outputs/ensemble_results.txt', 'a') as results_file:
                results_file.write(f'\nEpoch: {ep}: Ensemble accu-{ensemble_eval_accuracy[-1]}')

            with open('outputs/parallel_results.txt', 'a') as results_file:
                results_file.write(
                    f'\nEpoch: {ep}: Parallel loss-{parallel_eval_loss[-1]}. Accu. {parallel_eval_accuracy[-1]}')

            # If we are at a checkpoint epoch
            if ep + 1 in epochs:
                # Testing
                individual_corrects = len(data.mode) * [0]
                ensemble_correct = 0
                parallel_correct = 0
                # Iterate through dataloader
                with torch.no_grad():
                    for x, target in testing_loader:
                        # Mode-wise models (for individual and ensemble architectures)
                        # Get feature maps, avg, and flatten (just like in the whole model)
                        features = [model.flat(model.global_avg_pool(model.get_features(x[:, ch].squeeze(1))))
                                    for ch, model in enumerate(models)]

                        # Get final output for each model and do backprop
                        [optimizer.zero_grad() for optimizer in optimizers]
                        outs = [model(x[:, ch].squeeze(1)) for ch, model in enumerate(models)]
                        preds = [torch.round(out) for out in outs]
                        individual_corrects = [torch.sum(pred == target.unsqueeze(1)) for pred in preds]

                        # Determine ensemble prediciton and accuracy
                        pred = torch.round(torch.mean(torch.stack(outs, dim=0), dim=0))
                        ensemble_correct += torch.sum(pred == target.unsqueeze(1))

                        # Feed parallel-extracted features into full classifier
                        parallel_optimizer.zero_grad()
                        out = fetc_parallel_classifier(torch.stack(features, dim=1))
                        pred = torch.round(out)
                        parallel_correct = torch.sum(pred == target.unsqueeze(1))
                        loss = loss_fn(out, target.unsqueeze())
                        parallel_loss += loss.item()

                # Update validation checks
                individual_test_accuracies[-1] = [correct / len(test_set) for correct in individual_corrects]
                ensemble_test_accuracy[-1] = ensemble_correct / len(test_set)
                parallel_test_accuracy[-1] = parallel_correct / len(test_set)

                # Write outputs
                with open('outputs/individual_results.txt', 'a') as results_file:
                    results_file.write(f'\n>>>Epoch: {ep}:')
                    for mode, accu in zip(data.mode, individual_test_accuracies[-1]):
                        results_file.write(f'-- {mode}: Accu-{accu} -- ')
                    results_file.write(f'<<<\n')

                with open('outputs/ensemble_results.txt', 'a') as results_file:
                    results_file.write(f'\n>>>Epoch: {ep}: Ensemble accu-{ensemble_test_accuracy[-1]}<<<\n')

                with open('outputs/parallel_results.txt', 'a') as results_file:
                    results_file.write(
                        f'\n>>>Epoch: {ep}: Parallel Accu. {parallel_test_accuracy[-1]}<<<\n')


if __name__ == '__main__':
    main()
