import torch
from torch import nn


def masked_loss(loss_fn):
    def masked_loss_fn(outputs, targets, mask):
        loss = loss_fn(outputs.reshape(-1, 1, 1).expand(mask.shape),
                       targets.reshape(-1, 1, 1).expand(mask.shape))
        loss = loss * mask.float()
        loss = torch.sum(loss) / torch.sum(mask.float())
        return loss

    return masked_loss_fn


def train_epoch(model, loader, loss_fn, optimizer, masked_loss_fn=False):
    model.train()
    total_loss = 0
    if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
        model.cuda()
    for x, target in loader:
        if torch.cuda.is_available():
            x = x.cuda() if x.device.type != 'cuda' else x
            target = target.cuda() if target.device.type != 'cuda' else target
        optimizer.zero_grad()
        out = model(x).squeeze(1)
        if masked_loss_fn:
            loss = loss_fn(out, target.float(), torch.all((~torch.isnan(x)), dim=1))
        else:
            loss = loss_fn(out, target.float())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        # Clean up for memory
        del x, target, out, loss
        torch.cuda.empty_cache()

    epoch_loss = total_loss / len(loader)
    return epoch_loss


def valid_epoch(model, loader, loss_fn, masked_loss_fn=False):
    model.eval()
    if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
        model.cuda()
    with torch.inference_mode():
        correct, total_loss = 2 * [0]
        for x, target in loader:
            if torch.cuda.is_available():
                x = x.cuda() if x.device.type != 'cuda' else x
                target = target.cuda() if target.device.type != 'cuda' else target
            out = model(x).squeeze(1)
            if masked_loss_fn:
                loss = loss_fn(out, target.float(), torch.all((~torch.isnan(x)), dim=1))
            else:
                loss = loss_fn(out, target.float())
            total_loss += loss.item()
            prediction = torch.round(out)
            correct += (prediction == target).sum().item()

            # Clean up for memory
            del x, target, out, loss, prediction
            torch.cuda.empty_cache()

    val_loss = total_loss / len(loader)
    val_accu = correct / len(loader.dataset)
    return val_loss, val_accu


def test_model(model, loader):
    correct = 0
    model.eval()
    if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
        model.cuda()
    with torch.inference_mode():
        for x, target in loader:
            if torch.cuda.is_available():
                x = x.cuda() if x.device.type != 'cuda' else x
                target = target.cuda() if target.device.type != 'cuda' else target
            out = model(x).squeeze(1)
            prediction = torch.round(out)
            correct += (prediction == target).sum().item()

            # Clean up for memory
            del x, target, out, prediction
            torch.cuda.empty_cache()
    return correct


def trash_in(model, trash_heap_size=500):
    device = next(model.parameters()).device
    _in = torch.rand(trash_heap_size, *model.input_size, device=device)
    trash_out = []
    model.eval()
    for trash in _in:
        _out = model(trash.unsqueeze(0)).item()
        trash_out.append(_out)
    return trash_out


def fold_cross_validate(model_fn, data_folds,
                        epochs=125, learning_rate=0.01, batch_size=32,
                        loss_fn=nn.BCELoss(), masked_loss_fn=False, optimizer_fn=torch.optim.SGD, **optim_kwargs):
    accuracy = []
    running_loss = []
    models = []
    for fold, test_set in enumerate(data_folds):
        model = model_fn(test_set.dataset.shape)
        train_sets = [data_folds[index] for index in range(5) if index != fold]
        train_set = torch.utils.data.ConcatDataset(train_sets)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
        optimizer = optimizer_fn(model.parameters(), lr=learning_rate, **optim_kwargs)

        model.train()
        if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
            model.to(torch.device('cuda'))

        running_loss.append([])
        for epoch in range(epochs):
            model.train(True)
            loss = train_epoch(model, train_loader, loss_fn, optimizer, masked_loss_fn=masked_loss_fn)
            running_loss[fold].append(loss)
        model.eval()
        correct = test_model(model, test_loader)
        accuracy.append(100 * correct / len(test_loader.sampler))
        models.append(model)
        print(f'{model.name} accuracy for fold {fold + 1}: {accuracy[-1]:.2f}%')
    print(f'>>>{model.name} average accuracy: {sum(accuracy) / len(accuracy):.2f}%<<<')
    return accuracy, running_loss, models
