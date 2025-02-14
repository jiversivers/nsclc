import torch
import torchvision.transforms.v2 as tvt
from torch import nn

from my_modules.custom_models import AutoEncoderMLP
from my_modules.nsclc import NSCLCDataset, set_seed
import torch.multiprocessing as mp


def main():
    # Set random seed for reproducibility
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = NSCLCDataset('D:/Paola/JI-Finalized NSCLC Dataset Oct 2024',
                              ['fad', 'nadh', 'orr', 'shg', 'intensity'],
                              device=torch.device('cpu'), label='Mask', mask_on=False)
    train_data.to(device)
    train_data.transforms = tvt.Compose([tvt.RandomVerticalFlip(p=0.25),
                                         tvt.RandomHorizontalFlip(p=0.25),
                                         tvt.RandomRotation(degrees=(-180, 180))])

    eval_data = NSCLCDataset('NSCLC_Data_for_ML', ['fad', 'nadh', 'orr', 'shg', 'intensity'],
                             device=torch.device('cpu'), label='mask', mask_on=False)

    subsampler = [i for i in torch.utils.data.SubsetRandomSampler(range(0, len(train_data)))]
    train_indices = subsampler[:int(0.8 * len(subsampler))]
    eval_indices = subsampler[int(0.8 * len(subsampler)):]
    train_data = torch.utils.data.Subset(train_data, train_indices)
    eval_data = torch.utils.data.Subset(eval_data, eval_indices)

    model = AutoEncoderMLP(train_data.dataset.shape, train_data.dataset[0][1].shape, 512)
    model.to(device)

    batch_size = 32
    epochs = 500
    learning_rate = 0.001
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=len(eval_data), shuffle=False)

    results = 'results.txt'
    with open(results, 'w') as f:
        f.write('Results\n')

    train_loss = []
    eval_loss = []
    for ep in range(epochs):
        # Train
        for x, target in train_loader:
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        with open(results, 'a') as f:
            f.write(f'Epoch {ep}: Train loss {loss} -- ')

        # Eval
        with torch.no_grad():
            for x, target in eval_loader:
                x, target = x.to(device), target.to(device)
                output = model(x)
                loss = loss_function(output, target)
                eval_loss.append(loss.item())

        with open(results, 'a') as f:
            f.write(f'Eval loss {loss}.\n')

        if ep == 0 or loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), f'NSCLC_masking_model_best.pt')
            with open(results, 'a') as f:
                f.write(f'New best saved at epoch {ep}\n')
        if (ep + 1) % 50 == 0:
            torch.save(model.state_dict(), f'NSCLC_masking_model_{ep}.pth')
            with open(results, 'a') as f:
                f.write(f'Checkpoint saved at epoch {ep}\n')

if __name__ == '__main__':
    main()