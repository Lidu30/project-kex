import matplotlib.pyplot as plt 
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix as sk_confusion_matrix, accuracy_score, roc_curve, precision_recall_curve, auc
import os
import numpy as np


def plot_training_validation_curves(epoch_data, output_dir, run_identifier):
    """Plots training/validation loss, accuracy, and validation AUROC."""
    if not epoch_data:
        print(f"No epoch data provided for run {run_identifier} to plot training curves.")
        return

    epochs = range(1, len(epoch_data['train_loss']) + 1)
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Loss
    axs[0].plot(epochs, epoch_data['train_loss'], 'bo-', label='Training Loss')
    axs[0].plot(epochs, epoch_data['val_loss'], 'ro-', label='Validation Loss')
    axs[0].set_title(f'Training and Validation Loss (Run {run_identifier})')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Accuracy
    axs[1].plot(epochs, epoch_data['train_acc'], 'bo-', label='Training Accuracy')
    axs[1].plot(epochs, epoch_data['val_acc'], 'ro-', label='Validation Accuracy')
    axs[1].set_title(f'Training and Validation Accuracy (Run {run_identifier})')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].legend()
    axs[1].grid(True)

    # Validation AUROC
    axs[2].plot(epochs, epoch_data['val_auroc'], 'go-', label='Validation AUROC')
    axs[2].set_title(f'Validation AUROC (Run {run_identifier})')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('AUROC')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f'training_curves_run_{run_identifier}.png')
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Saved training curves to {plot_filename}")

def plot_boxplots(data_dict, title_prefix, output_dir, filename_suffix):
    """Generates box plots for a dictionary of metrics."""
    labels = list(data_dict.keys())
    data_values = [data_dict[key] for key in labels]

    plt.figure(figsize=(max(10, 2 * len(labels)), 6)) # Dynamic width
    plt.boxplot(data_values, labels=labels, patch_artist=True)
    plt.title(f'{title_prefix} Performance Distribution ({len(data_values[0])} Runs)')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f'{filename_suffix}_boxplots.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved {title_prefix.lower()} boxplots to {plot_filename}")

def plot_confusion_matrix_custom(y_true, y_pred, classes, title, output_dir, filename):
    """Plots a confusion matrix."""
    cm = sk_confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, filename)
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved confusion matrix to {plot_filename}")

def plot_roc_curve_custom(y_true, y_score, title, output_dir, filename):
    """Plots an ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, filename)
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved ROC curve to {plot_filename}")

def plot_precision_recall_curve_custom(y_true, y_score, title, output_dir, filename):
    """Plots a Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision) # Note: sklearn's average_precision_score is also good
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, filename)
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved Precision-Recall curve to {plot_filename}")

def plot_bar_chart_with_errors(mean_scores, std_devs, labels, title, output_dir, filename):
    """
    Generates a bar chart of mean scores with error bars for standard deviations.
    """
    if not all(isinstance(m, (int, float)) and isinstance(s, (int, float)) for m, s in zip(mean_scores, std_devs)):
        print(f"Skipping bar chart '{title}' due to non-numeric mean/std values.")
        return
    if len(mean_scores) != len(std_devs) or len(mean_scores) != len(labels):
        print(f"Skipping bar chart '{title}' due to mismatched lengths of means, stds, or labels.")
        return
        
    x = np.arange(len(labels))
    width = 0.4  # Width of the bars

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(labels)), 6)) # Dynamic width based on number of bars
    bars = ax.bar(x, mean_scores, width, label='Mean Score', yerr=std_devs, capsize=5, ecolor='gray', alpha=0.8)

    # Add labels, title, ticks, etc.
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(bottom=max(0, min(mean_scores) - max(std_devs) - 0.1), top=min(1.05, max(mean_scores) + max(std_devs) + 0.1)) # Adjust y-limits dynamically
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add exact mean value labels on top of bars
    ax.bar_label(bars, fmt='%.3f', padding=3) # Uses matplotlib's built-in bar label function

    fig.tight_layout() # Adjust layout
    plot_filename = os.path.join(output_dir, filename)
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Saved bar chart to {plot_filename}")