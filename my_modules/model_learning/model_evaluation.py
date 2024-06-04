import torch
from matplotlib import pyplot as plt


def calculate_auc_roc(model, loader, print_results=False, make_plot=False,
                      device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    model.eval()
    outs = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)
    with torch.no_grad():
        for x, target in loader:
            outs = torch.cat((outs, model(x)), dim=0)
            targets = torch.cat((targets, target), dim=0)
        thresholds, idx = torch.sort(outs.detach().squeeze())
        sorted_targets = targets[idx]
        tpr = []
        fpr = []
        acc = []
        for t in thresholds:
            positive_preds = thresholds >= t
            true_positives = torch.logical_and(positive_preds, sorted_targets == 1)
            true_negatives = torch.logical_and(~positive_preds, sorted_targets == 0)
            false_positives = torch.logical_and(positive_preds, sorted_targets == 0)
            false_negatives = torch.logical_and(~positive_preds, sorted_targets == 1)
            tpr.append((torch.sum(true_positives) / (torch.sum(sorted_targets == 1))).item())
            fpr.append((torch.sum(false_positives) / (torch.sum(sorted_targets == 0))).item())
            acc.append((torch.sum(true_positives) + torch.sum(true_negatives)) / len(sorted_targets))
        tpr.append(0.0)
        fpr.append(0.0)
        d_fpr = [f1 - f0 for f1, f0 in zip(fpr[:-1], fpr[1:])]
        a = [t * df for df, t in zip(d_fpr, tpr[:-1])]

    # Final metrics
    auc = sum(a)
    best_acc = max(acc)
    thresh = thresholds[acc.index(best_acc)]

    # Optional print of metrics
    if print_results:
        print(f'>>> AUC-ROC {auc:.2f} || '
              f'Best accuracy of {best_acc:.2f} at threshold of {thresh:.2f} <<<')

    # Optional figure creation
    if make_plot:
        # Move to host memory (if on GPU)
        fpr = fpr.cpu()
        tpr = tpr.cpu()
        thresholds = thresholds.cpu()
        acc = acc.cpu()

        # For plot
        fig, ax1 = plt.subplots()

        # ROC
        ax1.plot(fpr, tpr, 'r-', label='ROC')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_ylim([0, 1])
        ax1.tick_params(axis='both', labelcolor='r')

        # Accuracy
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([0, 1])
        ax2.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twiny()
        ax2.plot(thresholds, acc, 'b-', label='Accuracy')
        ax2.set_xlabel('Threshold')
        ax2.tick_params(axis='x', labelcolor='b')

        # Legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='lower right')

        return auc, best_acc, thresh, fig
    else:
        return auc, best_acc, thresh
