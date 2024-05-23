import torch
from pretrainedmodels import inceptionresnetv2
import torch.multiprocessing as mp

from my_modules.model_learning.loader_maker import fold_augmented_data
from my_modules.nsclc import NSCLCDataset
from my_modules.custom_models import RegularizedMLPNetWithPretrainedFeatureExtractor as Classifier


def main():
    # Set up multiprocessing context
    mp.set_start_method('forkserver', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define our base feature extractor and turn the gradients off -- we won't train it, just use it to feed our MLP.
    feature_extractor = inceptionresnetv2(num_classes=1001, pretrained=False)

    # Load pretrained from download
    feature_extractor.load_state_dict(torch.load(r'/home/jdivers/data/torch_checkpoints/pretrained_models/'))
    for params in feature_extractor.parameters():
        params.requires_grad = False

    # Set up the dataset for this model
    # Images, no mask (feature extractor will hopefully handle this), normalized (already is),
    data = NSCLCDataset('data/NSCLC_Data_for_ML', ['orr', 'taumean', 'boundfraction'], device='cpu',
                        label='Metastases', mask_on=False)
    print('Normalizing data to channel max...')
    data.augment()
    data.normalize_channels_to_max()

    batch_size = 21
    learning_rate = 0.0005
    optimizer_fn = torch.optim.RMSprop
    epochs = 300
    loss_fn = torch.nn.CrossEntropyLoss()

    # Check that shapes are as expected
    m = Classifier(data.shape, feature_extractor.features)
    out = m(data[0][0].unsqueeze(0))
    print(f'Input shape {data[0][0].unsqueeze(0)}')
    print(f'After feature extraction: {m.get_features(data[0][0].unsqueeze(0)).shape}')
    print(f'After classifier: {out.shape}')

    models = []
    accuracy = []
    running_loss = []

    data_folds = fold_augmented_data(data, num_folds=5, augmentation_factor=5)

    for fold, test_set in enumerate(data_folds):
        print(f'Fold {fold + 1}\n________________________________')
        # Make model, loaders, & optimizer for fold
        model = Classifier(data.shape, feature_extractor.features)
        model.to(device)
        train_sets = [data_folds[index] for index in range(5) if index != fold]
        train_set = torch.utils.data.ConcatDataset(train_sets)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
        optimizer = optimizer_fn(model.parameters(), lr=learning_rate)

        # Train
        model.train()
        for ep in range(epochs):
            print(f'Epoch: {ep + 1}')
            for x, target in train_loader:
                if torch.cuda.is_available() and not x.is_cuda:
                    x = x.cuda()
                if torch.cuda.is_available() and not target.is_cuda:
                    target = target.cuda()
                # Extract features
                out = model(x)

                # Array to reform target to look like out
                y = torch.zeros_like(out)

                # Put 100% certainty on the class
                for r, t in enumerate(target):
                    y[r, t] = 1

                loss = loss_fn(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss.append(loss.item())
        # Test
        correct = 0
        for x, target in test_loader:
            out = model(x)
            pred = torch.argmax(out)
            correct += 1 if pred == target.item() else 0
            accuracy.append(100 * correct / len(test_loader.sampler))
            models.append(model)
            print(f'Inception + MLP accuracy for fold {fold + 1}: {accuracy[-1]:.2f}%')
    print(f'>>>Inception + MLP average accuracy: {sum(accuracy) / len(accuracy):.2f}%<<<')


if __name__ == '__main__':
    main()
