from collections import OrderedDict
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, average_precision_score, roc_curve, auc, \
    accuracy_score, balanced_accuracy_score, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay


def calculate_auc_roc(model, loader, print_results=False, make_plot=False,
                      device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    model.eval()
    outs = []
    targets = []
    with torch.no_grad():
        for x, target in loader:
            outs.append(model(x).tolist())
            targets.append(target.tolist())
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
            acc.append(((torch.sum(true_positives) + torch.sum(true_negatives)) / len(sorted_targets)).item())
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
        thresholds = thresholds.cpu()

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


def score_model(model, loader, print_results=False, make_plot=False, threshold_type='none'):
    def make_the_plots():
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=scores['ROC-AUC']).plot(ax=ax1)
        ax1.set_title('ROC')
        precision, recall, _ = precision_recall_curve(targets, outs)
        PrecisionRecallDisplay.from_predictions(targets, outs).plot(ax=ax2)
        ax2.set_title('Precision Recall Curve')
        ConfusionMatrixDisplay.from_predictions(targets, preds).plot(ax=ax3)
        ax3.set_title('Confusion Matrix')
        return fig

    scores = OrderedDict()
    model.eval()
    outs = []
    targets = []
    with torch.no_grad():
        for x, target in loader:
            outs.append(model(x).tolist())
            targets.append(target.tolist())
    # ROC
    fpr, tpr, thresholds = roc_curve(targets, outs, pos_label=1)
    scores['ROC-AUC'] = auc(fpr, tpr)
    scores['Optimal Threshold from ROC'] = thresholds[np.argmax(tpr - fpr)]

    # Precision-Recall
    precision, recall, thresholds = precision_recall_curve(targets, outs)
    scores['F1 Score'] = np.nanmax((2 * precision * recall) / (precision + recall))
    scores['Optimal Threshold from F1'] = thresholds[np.argmax(scores['F1 Score'])]
    scores['Average Precision'] = average_precision_score(targets, outs)

    # Now use threshold to make predictions and score
    preds = torch.zeros_like(outs)
    match threshold_type.lower():
        case 'roc':
            preds[outs > scores['Optimal Threshold from ROC']] = 1
            scores['Accuracy at Threshold'] = accuracy_score(targets, preds)
            scores['Balanced Accuracy at Threshold'] = balanced_accuracy_score(targets, preds)
            scores['Confusion Matrix'] = confusion_matrix(targets, preds)
            if make_plot:
                fig_out = make_the_plots()
        case 'f1':
            preds[outs > scores['Optimal Threshold from F1']] = 1
            scores['Accuracy at Threshold'] = accuracy_score(targets, preds)
            scores['Balanced Accuracy at Threshold'] = balanced_accuracy_score(targets, preds)
            scores['Confusion Matrix'] = confusion_matrix(targets, preds)
            if make_plot:
                fig_out = make_the_plots()
        case 'none' | 'both':
            fig_out = {}
            # ROC
            preds[outs > scores['Optimal Threshold from ROC']] = 1
            scores['Accuracy at ROC Threshold'] = accuracy_score(targets, preds)
            scores['Balanced Accuracy at ROC Threshold'] = balanced_accuracy_score(targets, preds)
            scores['Confusion Matrix from ROC Threshold'] = confusion_matrix(targets, preds)
            if make_plot:
                fig_out['ROC Threshold'] = make_the_plots()
            # F1
            preds[outs > scores['Optimal Threshold from F1']] = 1
            scores['Accuracy at F1 Threshold'] = accuracy_score(targets, preds)
            scores['Balanced Accuracy at F1 Threshold'] = balanced_accuracy_score(targets, preds)
            scores['Confusion Matrix from F1 Threshold'] = confusion_matrix(targets, preds)
            if make_plot:
                fig_out['F1 Threshold'] = make_the_plots()
        case _:
            raise ValueError(f'Unrecognized threshold type: {threshold_type}. '
                             f'Accepted thresholds are ROC, F1, or both (default).')

    if print_results:
        print('_____________________________________________________')
        for key, item in scores.items():
            if 'Confusion' not in key:
                print(f'|\t{key:<35} {f'{item:.4f}':>10}\t|')
        print('_____________________________________________________')

    if make_plot:
        return scores, fig_out
    else:
        return scores
