import os

import torch
from pretrainedmodels import xception
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

from my_modules.model_learning.loader_maker import split_augmented_data
from my_modules.nsclc import NSCLCDataset
from my_modules.custom_models import RegularizedMLPNet as RegMLP, FeatureExtractorToClassifier as FETC


def main():
    # Set up multiprocessing context
    mp.set_start_method('forkserver', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prep output dirs and files
    os.makedirs('outputs/plots', exist_ok=True)
    with open('outputs/results.txt', 'w') as results_file:
        results_file.write('Results')

    # Define our base feature extractor and turn the gradients off -- we won't train it, just use it to feed our MLP.
    feature_extractor = xception(num_classes=1000, pretrained=False)

    # Load pretrained from download
    state_dict = torch.load(r'/home/jdivers/data/torch_checkpoints/pretrained_models/xception-43020ad28.pth')
    state_dict['last_linear.weight'] = state_dict.pop('fc.weight')
    state_dict['last_linear.bias'] = state_dict.pop('fc.bias')
    feature_extractor.load_state_dict(state_dict)
    for params in feature_extractor.parameters():
        params.requires_grad = False

    # Set up the dataset for this model
    # Images, no mask (feature extractor will hopefully handle this), normalized (already is),
    data = NSCLCDataset('data/NSCLC_Data_for_ML', ['orr', 'taumean', 'boundfraction'], device='cpu',
                        label='Metastases', mask_on=False)
    print('Normalizing data to channel max...')
    data.augment()
    data.normalize_channels_to_max()

    batch_size = 64
    learning_rates = [5e-5, 1e-5, 5e-6, 1e-6]
    optimizer_fns = {'Adam': [torch.optim.Adam, {}]}
    epochs = [5, 10, 50, 125, 250, 500, 1000, 2500]
    loss_fn = torch.nn.CrossEntropyLoss()

    # Define base classifier
    classifier = RegMLP

    # Prepare data loaders
    train_set, eval_set, test_set = split_augmented_data(data, augmentation_factor=5, split=(0.75, 0.15, 0.1))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                               drop_last=(True if len(train_set) % batch_size == 1 else False))
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size, shuffle=False, num_workers=0,
                                              drop_last=(True if len(eval_set) % batch_size == 1 else False))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,
                                              drop_last=(True if len(test_set) % batch_size == 1 else False))

    training_loss = []
    evaluation_loss = []
    evaluation_accuracy = []
    testing_accuracy = []
    for lr in learning_rates:
        # Make full model
        model = FETC(data.shape, feature_extractor=feature_extractor, classifier=classifier,
                     layer='conv4')
        model.to(device)

        # Make optimizer at the current larning rate with only classifier parameters
        optimizer = optimizer_fns['Adam'][0](model.classifier.parameters(), lr=lr)

        # Nest iteration lists for tracking model learning
        training_loss.append([])
        evaluation_accuracy.append([])
        evaluation_loss.append([])
        testing_accuracy.append([])

        # Find max number of epochs to consider and do that, checking along the way for others
        epoch = sorted(epochs)[-1]
        for ep in range(epoch):
            # Training
            epoch_loss = 0
            model.train()
            for x, target in train_loader:
                if torch.cuda.is_available() and not x.is_cuda:
                    x = x.cuda()
                if torch.cuda.is_available() and not target.is_cuda:
                    target = target.cuda()

                optimizer.zero_grad()
                out = model(x)

                loss = loss_fn(out, target.unsqueeze(1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            training_loss[-1].append(epoch_loss / len(train_set))

            # Validation
            model.eval()
            eval_loss = 0
            correct = 0
            with torch.no_grad():
                for x, target in eval_loader:
                    if torch.cuda.is_available() and not x.is_cuda:
                        x = x.cuda()
                    if torch.cuda.is_available() and not target.is_cuda:
                        target = target.cuda()
                    out = model(x)

                    # Change target label from binary label to one-hot 2-bit vector
                    y = torch.zeros_like(out)  # Array to reform target to look like out
                    for r, t in enumerate(target):  # Put 100% certainty on the class
                        y[r, t.long()] = 1.

                    loss = loss_fn(out, y)
                    eval_loss += loss.item()
                    pred = torch.argmax(out, dim=1)
                    correct += torch.sum(pred == target).item()
                evaluation_loss[-1].append(eval_loss / len(eval_set))
                evaluation_accuracy[-1].append(100 * correct / len(eval_loader.sampler))

            with open('outputs/results.txt', 'a') as results_file:
                results_file.write(f'\nEpoch {ep + 1}: Train.Loss: {training_loss[-1][-1]:.4f}, '
                                   f'Eval.Loss: {evaluation_loss[-1][-1]:.4f}.'
                                   f'Eval.Accu: {evaluation_accuracy[-1][-1]:.2f}%')

            # See if we are at one of our training length checkpoints. Save and test if we are
            if ep + 1 in epochs:
                # save trained model at this many epochs
                torch.save(model.state_dict(),
                           f'xception_features_to_regMLP_{ep + 1}-Epochs_{lr}-LearningRate.pth')

                plt.plot(training_loss[-1])
                plt.plot(evaluation_loss[-1])
                plt.savefig(f'output/plots/loss_xception_features_to_regMLP_{ep + 1}-Epochs_{lr}-LearningRate.png')
                plt.clf()

                # Test
                correct = 0
                model.eval()
                with torch.no_grad():
                    for x, target in test_loader:
                        if torch.cuda.is_available() and not x.is_cuda:
                            x = x.cuda()
                        if torch.cuda.is_available() and not target.is_cuda:
                            target = target.cuda()
                        out = model(x)
                        pred = torch.argmax(out, dim=1)
                        correct += torch.sum(pred == target).item()
                    testing_accuracy[-1].append(100 * correct / len(test_loader.sampler))
                with open('outputs/results.txt', 'a') as results_file:
                    results_file.write(f'\n>>>Test.accu at {ep + 1} epochs with learning rate of {lr}: '
                                       f'{testing_accuracy[-1][-1]}<<<\n')


if __name__ == '__main__':
    main()
