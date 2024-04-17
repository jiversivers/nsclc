import torch


def masked_loss(loss_fn, predictions, targets, image):
    mask = (~torch.isnan(image)).float()
    loss = loss_fn(predictions.float(), targets.float())
    masked = mask * loss
    masked = torch.sum(masked) / torch.sum(mask)  # Loss averaged over pixels that were numeric
    return masked


def train_epoch(model, loader, loss_fun, optimizer):
    model.train()
    total_loss = 0
    if torch.cuda.is_available():
        model.cuda()
    for x, target in loader:
        if torch.cuda.is_available():
            x, target = x.cuda(), target.cuda()
        optimizer.zero_grad()
        out = model(x).squeeze(1)
        loss = masked_loss(loss_fun, out, target, x)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    epoch_loss = total_loss / len(loader.dataset)
    return epoch_loss


def valid_epoch(model, loader, loss_fun):
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    with torch.no_grad():
        correct, total_loss = 2 * [0]
        for x, target in loader:
            if torch.cuda.is_available():
                x, target = x.cuda(), target.cuda()
            out = model(x).squeeze(1)
            loss = masked_loss(loss_fun, out, target, x)
            total_loss += loss.item()
            prediction = torch.round(out)
            print(f'Out: {out}. Pred: {prediction}. Targ: {target}.')
            correct += (prediction == target).sum().item()
            print(f'Accu: {correct / len(out)}')
    val_loss = total_loss / len(loader)
    val_accu = correct / len(loader.dataset)
    return val_loss, val_accu


def test_model(model, loader):
    correct = 0
    model.eval()
    with torch.no_grad():
        for x, target in loader:
            if torch.cuda.is_available():
                x, target = x.cuda(), target.cuda()
            out = model(x).squeeze(1)
            prediction = torch.round(out)
            correct += (prediction == target).sum().item()
    return correct
