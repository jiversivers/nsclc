import torch


def masked_loss(loss_fn, predictions, targets, image):
    mask = (~torch.isnan(image)).float()
    loss = loss_fn(predictions, targets)
    masked = mask * loss
    masked = torch.sum(masked) / torch.sum(mask)  # Loss averaged over pixels that were numeric
    return masked


def train_epoch(model, loader, loss_fun, optimizer):
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
        loss = loss_fun(out.float(), target.float())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    epoch_loss = total_loss / len(loader)
    return epoch_loss


def valid_epoch(model, loader, loss_fun):
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
            loss = loss_fun(out.float(), target.float())
            total_loss += loss.item()
            prediction = torch.round(out)
            correct += (prediction == target).sum().item()
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
    return correct
