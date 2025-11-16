import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt # Removed plotting for simplicity, can be added back if needed
import os
#from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
# from sklearn import metrics # Redundant import
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix as sk_confusion_matrix, accuracy_score, roc_curve, precision_recall_curve, auc
import argparse
import numpy as np
from kan import *
# Import the updated data loading function from finaldata.py
from channelpca import create_schizophrenia_datasets # Use finaldata which includes validation split
from plot_file import plot_training_validation_curves, plot_boxplots, plot_confusion_matrix_custom, plot_roc_curve_custom, plot_precision_recall_curve_custom, plot_bar_chart_with_errors


# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Schizophrenia EEG Classification with Repeated Splits and Validation')
parser.add_argument('--healthy_dir', type=str, required=True, help='Directory with healthy subject EEG files')
parser.add_argument('--schizophrenia_dir', type=str, required=True, help='Directory with schizophrenia subject EEG files')
parser.add_argument('--output_dir', type=str, default='./Results/Schizophrenia_ValidatedRuns/', help='Base directory for results')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training, validation, and testing') # Consistent default with split.py
parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs per run') # Consistent default with split.py
parser.add_argument('--hidden_size', type=int, default=40, help='Hidden layer size in KAN') # Consistent default with split.py
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for AdamW')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for AdamW')
# parser.add_argument('--load', type=str, default='', help='Path to load a pre-trained model (not typically used in multi-run setup)') # Removed load option for multi-run setup
parser.add_argument('--grid_size', type=int, default=8, help='Grid size for KAN splines') # Consistent default with split.py
parser.add_argument('--spline_order', type=int, default=4, help='Spline order for KAN')
parser.add_argument('--num_runs', type=int, default=10, help='Number of repeated train/val/test splits') # Added from split.py
parser.add_argument('--test_split_ratio', type=float, default=0.2, help='Proportion of subjects for the initial test set') # Added from split.py
parser.add_argument('--validation_ratio', type=float, default=0.2, help='Proportion of the *remaining* training data to use for validation (e.g., 0.25 means 25% of the (1-test_split_ratio) data)') # Added validation split ratio
args = parser.parse_args()


class SchizophreniaKANModel(nn.Module): #wraps the kan model in Pytorch nn.module or the class inherits from the nn module

    # _init_ a constractor that run when we create an instance of the class*

    def __init__(self, input_size, hidden_size, num_classes, grid_size=8, spline_order=4, seed=0): # Default values match KAN defaults
        super(SchizophreniaKANModel, self).__init__()
        # Calls the parent class constructor so that pytorch kan track the parameters and modules
        self.model = KAN(
            width=[input_size, hidden_size, hidden_size*2, hidden_size, num_classes], # Example: 3 hidden layers like test.py
            grid=grid_size,
            k=spline_order,
            seed=seed, # Base seed for KAN internal randomness
        )
        self.model = self.model.speed() # Speed up KAN

    #Any subclass of the nn.Module must inherit the forward() method
    #to define how input data flows through the model
    def forward(self, x):
        return self.model(x)

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = sk_confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return specificity


# --- Helper Function for Evaluation ---
def evaluate_model(model, dataloader, criterion, device):
    # This puts the model in evaluation mode and things like dropout and batch normalization updates.
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    # disables automatic gradient tracking 
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            _, predicted = output.max(1)
            probs = output.softmax(dim=1)[:, 1] # Probability of class 1

            #the following lines append all the results from all batches inti the lists
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_preds)
    auroc = roc_auc_score(all_targets, all_probs)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    return avg_loss, accuracy, auroc, precision, recall, f1, all_targets, all_preds, all_probs


# --- Main Execution ---
if __name__ == "__main__":

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be saved in: {args.output_dir}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate seeds for each run (like split.py)
    seed_generator = np.random.default_rng(42) # We can use a fixed seed for the generator itself for reproducibility of runs
    #size=args.num_runs
    run_seeds = seed_generator.integers(0, 2**32 - 1, size=args.num_runs)

    # Lists to store metrics from each run's *test* set evaluation
    all_test_segment_accs = []
    all_test_segment_aurocs = []
    all_test_segment_precisions = []
    all_test_segment_recalls = []
    all_test_segment_f1s = []
    all_test_segment_specificities = []

    all_test_subject_accs = []
    all_test_subject_aurocs = []
    all_test_subject_precisions = []
    all_test_subject_recalls = []
    all_test_subject_f1s = []
    all_test_subject_specificities = []

    #This list will store dictionaries, one for each run
    run_data_for_plotting = []


    # --- Loop for Repeated Runs ---
    for run_idx in range(args.num_runs):
        current_seed = int(run_seeds[run_idx]) # Ensure seed is standard int
        print(f"\n--- Starting Run {run_idx + 1}/{args.num_runs} (Seed: {current_seed}) ---")

        # Set seeds for reproducibility for this specific run
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)
            # Optional: uncomment for more deterministic behavior, might slow down training
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

        # --- 1. Load and prepare datasets for the current run/seed ---
        print("Loading and preparing datasets...")
        # Use finaldata.create_schizophrenia_datasets which yields train, val, and test loaders
       
        train_loader, val_loader, test_loader, input_size, n_classes, n_test_subjects, n_segments_per_subject = create_schizophrenia_datasets(
            args.healthy_dir,
            args.schizophrenia_dir,
            test_size=args.test_split_ratio,
            validation_ratio=args.validation_ratio, # Pass the validation ratio
            batch_size=args.batch_size,
            random_state=current_seed # Pass the current seed for splitting
        )
        
        print(f"Run {run_idx + 1}: Input size={input_size}, Num Classes={n_classes}")
        print(f"Run {run_idx + 1}: Num Test Subjects={n_test_subjects}, Segments/Subject={n_segments_per_subject}")

        # --- 2. Initialize the Model for the current run ---
        model = SchizophreniaKANModel(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_classes=n_classes,
            grid_size=args.grid_size,
            spline_order=args.spline_order,
            seed=current_seed # Pass seed to KAN
        ).to(device)


        # --- 3. Define optimizer, scheduler, and loss function ---
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        # Scheduler step can be based on validation loss
        
        #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        criterion = nn.CrossEntropyLoss() # cross entropy is a function from torch that measures how wrong the predisctions are


        # --- 4. Training Loop with Validation ---
        print(f"Starting training for run {run_idx + 1}...")
        best_val_auroc = -float('inf')
        best_epoch = -1
        best_model_state = None

        current_run_epoch_metrics = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [], 'val_auroc': []
        }

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()  #Clears old gradients from the last step.
                output = model(data)
                loss = criterion(output, target)
                #Computes gradients of the loss with respect to all model parameters.
                loss.backward()
                #Updates the model's parameters using the gradients computed by .backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()

                pbar.set_postfix({
                    'loss': f"{train_loss / (batch_idx + 1):.4f}",
                    'acc': f"{100. * train_correct / train_total:.2f}%"
                })
            pbar.close()
            train_epoch_loss = train_loss / len(train_loader)
            train_epoch_acc = 100. * train_correct / train_total

            # --- Validation Step ---
            val_loss, val_acc, val_auroc, _, _, _, _, _, _ = evaluate_model(model, val_loader, criterion, device)
            val_acc *= 100 # Convert to percentage for display

            print(f"  Epoch {epoch+1}/{args.epochs} - Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUROC: {val_auroc:.4f}")
            
            current_run_epoch_metrics['train_loss'].append(train_epoch_loss)
            current_run_epoch_metrics['train_acc'].append(train_epoch_acc) # Storing as percentage
            current_run_epoch_metrics['val_loss'].append(val_loss)
            current_run_epoch_metrics['val_acc'].append(val_acc) # Storing as percentage
            current_run_epoch_metrics['val_auroc'].append(val_auroc)

            # --- Learning Rate Scheduler Step ---
            #scheduler.step()
            scheduler.step(val_loss) # Step scheduler based on validation loss

            # --- Best Model Check ---
            if val_auroc > best_val_auroc:
                print(f"  ** New best validation AUROC: {val_auroc:.4f} (Epoch {epoch+1}). Saving model state. **")
                best_val_auroc = val_auroc
                best_epoch = epoch + 1
                best_model_state = model.state_dict()

        # --- End of Training for Run ---
        print(f"Finished training for run {run_idx + 1}. Best auroc {best_val_auroc} achieved at epoch {best_epoch}.")

        # --- 5. Final Evaluation on Test Set using Best Model ---
        
        print(f"Loading best model from epoch {best_epoch} for final test evaluation...")
        # Create a new model instance and load the best state to avoid issues with KAN internal state if any
        final_model = SchizophreniaKANModel(
            input_size=input_size, hidden_size=args.hidden_size, num_classes=n_classes,
            grid_size=args.grid_size, spline_order=args.spline_order, seed=current_seed
        ).to(device)
        final_model.load_state_dict(best_model_state)
        

        print(f"Starting final test evaluation for run {run_idx + 1}...")
        test_loss, test_segment_acc, test_segment_auroc, test_segment_precision, test_segment_recall, test_segment_f1, \
        all_test_targets, all_test_preds, all_test_probs = evaluate_model(final_model, test_loader, criterion, device)

        test_segment_specificity = calculate_specificity(all_test_targets, all_test_preds)
        all_test_segment_specificities.append(test_segment_specificity)

        print("\nTest Set Segment-Level Metrics (Run {}):".format(run_idx + 1))
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_segment_acc:.4f}")
        print(f"  AUROC: {test_segment_auroc:.4f}")
        print(f"  Precision: {test_segment_precision:.4f}")
        print(f"  Recall: {test_segment_recall:.4f}")
        #print(f"  F1 Score: {test_segment_f1:.4f}")
        #print(f"  Specificity: {test_segment_specificity:.4f}")
        # Append segment metrics to lists
        all_test_segment_accs.append(test_segment_acc)
        all_test_segment_aurocs.append(test_segment_auroc)
        all_test_segment_precisions.append(test_segment_precision)
        all_test_segment_recalls.append(test_segment_recall)
        all_test_segment_f1s.append(test_segment_f1)

        # --- Estimate Subject-Level Metrics for the run (similar to split.py) ---
       
        subject_preds_agg = []
        subject_targets_agg = []
        subject_probs_agg = []

        # Reconstruct subject groups from flat lists
        for i in range(0, len(all_test_targets), n_segments_per_subject):
            segment_targets = all_test_targets[i : i + n_segments_per_subject]
            segment_preds = all_test_preds[i : i + n_segments_per_subject]
            segment_probs = all_test_probs[i : i + n_segments_per_subject]

            if not segment_targets: continue # Skip if group is empty

            true_label = segment_targets[0] # there is a consistent label within group
            subject_targets_agg.append(true_label)

            # Majority voting for prediction
            pos_count = sum(p == 1 for p in segment_preds)
            neg_count = len(segment_preds) - pos_count
            final_pred = 1 if pos_count > neg_count else 0
            subject_preds_agg.append(final_pred)

            # Average probability for AUROC
            avg_prob = sum(segment_probs) / len(segment_probs)
            subject_probs_agg.append(avg_prob)

           
        run_subject_acc = accuracy_score(subject_targets_agg, subject_preds_agg)
        run_subject_auroc = roc_auc_score(subject_targets_agg, subject_probs_agg)
        run_subject_precision = precision_score(subject_targets_agg, subject_preds_agg, zero_division=0)
        run_subject_recall = recall_score(subject_targets_agg, subject_preds_agg, zero_division=0)
        run_subject_f1 = f1_score(subject_targets_agg, subject_preds_agg, zero_division=0)
        run_subject_specificity = calculate_specificity(subject_targets_agg, subject_preds_agg)
        

        print("\nTest Set Subject-Level Metrics (Run {} - Estimated):".format(run_idx + 1))
        print(f"  Accuracy: {run_subject_acc:.4f}")
        print(f"  AUROC: {run_subject_auroc:.4f}")
        print(f"  Precision: {run_subject_precision:.4f}")
        print(f"  Recall: {run_subject_recall:.4f}")
        #print(f"  F1 Score: {run_subject_f1:.4f}")
        #print(f"  Specificity: {run_subject_specificity:.4f}") 

        # Append subject metrics to lists
        all_test_subject_accs.append(run_subject_acc)
        all_test_subject_aurocs.append(run_subject_auroc)
        all_test_subject_precisions.append(run_subject_precision)
        all_test_subject_recalls.append(run_subject_recall)
        all_test_subject_f1s.append(run_subject_f1)
        all_test_subject_specificities.append(run_subject_specificity)
        
        run_data_for_plotting.append({
            'run_idx': run_idx + 1,
            'seed': current_seed,
            'epoch_metrics': dict(current_run_epoch_metrics), # Store a copy
            'segment_test_auroc': test_segment_auroc, # For potential other criteria
            'subject_test_auroc': run_subject_auroc,
            
            'segment_accuracy': test_segment_acc, # Store segment accuracy for this run
            'subject_accuracy': run_subject_acc,

            'segment_targets': list(all_test_targets),
            'segment_preds': list(all_test_preds),
            'segment_probs': list(all_test_probs),
            'subject_targets': list(subject_targets_agg),
            'subject_preds': list(subject_preds_agg),
            'subject_probs': list(subject_probs_agg)
        })
        
        #Optional: Save the best model for this run if needed
        run_output_dir = os.path.join(args.output_dir, f'run_{run_idx+1}_seed_{current_seed}')
        os.makedirs(run_output_dir, exist_ok=True)
        if best_model_state:
            torch.save(best_model_state, os.path.join(run_output_dir, 'best_val_model.pth'))


    # --- Final Summary Across All Runs ---
    print(f"\n--- Summary Across {args.num_runs} Runs (Test Set Performance) ---")

    print(f"  Accuracy:  {np.nanmean(all_test_segment_accs):.4f} +/- {np.nanstd(all_test_segment_accs):.4f}")
    print(f"  AUROC:     {np.nanmean(all_test_segment_aurocs):.4f} +/- {np.nanstd(all_test_segment_aurocs):.4f}")
    print(f"  Precision: {np.nanmean(all_test_segment_precisions):.4f} +/- {np.nanstd(all_test_segment_precisions):.4f}")
    print(f"  Recall:    {np.nanmean(all_test_segment_recalls):.4f} +/- {np.nanstd(all_test_segment_recalls):.4f}")
    #print(f"  F1 Score:  {np.nanmean(all_test_segment_f1s):.4f} +/- {np.nanstd(all_test_segment_f1s):.4f}")
    #print(f"  Specificity: {np.nanmean(all_test_segment_specificities):.4f} +/- {np.nanstd(all_test_segment_specificities):.4f}")

    print("\nSubject-Level Performance (Mean +/- Std Dev - Estimated):")
    print(f"  Accuracy:  {np.nanmean(all_test_subject_accs):.4f} +/- {np.nanstd(all_test_subject_accs):.4f}")
    print(f"  AUROC:     {np.nanmean(all_test_subject_aurocs):.4f} +/- {np.nanstd(all_test_subject_aurocs):.4f}")
    print(f"  Precision: {np.nanmean(all_test_subject_precisions):.4f} +/- {np.nanstd(all_test_subject_precisions):.4f}")
    print(f"  Recall:    {np.nanmean(all_test_subject_recalls):.4f} +/- {np.nanstd(all_test_subject_recalls):.4f}")
   # print(f"  Specificity: {np.nanmean(all_test_subject_specificities):.4f} +/- {np.nanstd(all_test_subject_specificities):.4f}")

    # Save summary results to a file
    summary_file = os.path.join(args.output_dir, 'summary_metrics_test_set.txt')
    with open(summary_file, 'w') as f:
        f.write(f"--- Summary Across {args.num_runs} Runs (Test Set Performance) ---\n\n")
        f.write(f"Parameters: {vars(args)}\n\n") # Log parameters used
        f.write("Segment-Level Performance (Mean +/- Std Dev):\n")
        f.write(f"  Accuracy:  {np.nanmean(all_test_segment_accs):.4f} +/- {np.nanstd(all_test_segment_accs):.4f}\n")
        f.write(f"  AUROC:     {np.nanmean(all_test_segment_aurocs):.4f} +/- {np.nanstd(all_test_segment_aurocs):.4f}\n")
        f.write(f"  Precision: {np.nanmean(all_test_segment_precisions):.4f} +/- {np.nanstd(all_test_segment_precisions):.4f}\n")
        f.write(f"  Recall:    {np.nanmean(all_test_segment_recalls):.4f} +/- {np.nanstd(all_test_segment_recalls):.4f}\n")
        #f.write(f"  F1 Score:  {np.nanmean(all_test_segment_f1s):.4f} +/- {np.nanstd(all_test_segment_f1s):.4f}\n\n")
        #f.write(f"  Specificity: {np.nanmean(all_test_segment_specificities):.4f} +/- {np.nanstd(all_test_segment_specificities):.4f}\n\n")
        f.write("Subject-Level Performance (Mean +/- Std Dev - Estimated):\n")
        f.write(f"  Accuracy:  {np.nanmean(all_test_subject_accs):.4f} +/- {np.nanstd(all_test_subject_accs):.4f}\n")
        f.write(f"  AUROC:     {np.nanmean(all_test_subject_aurocs):.4f} +/- {np.nanstd(all_test_subject_aurocs):.4f}\n")
        f.write(f"  Precision: {np.nanmean(all_test_subject_precisions):.4f} +/- {np.nanstd(all_test_subject_precisions):.4f}\n")
        f.write(f"  Recall:    {np.nanmean(all_test_subject_recalls):.4f} +/- {np.nanstd(all_test_subject_recalls):.4f}\n")
        #f.write(f"  F1 Score:  {np.nanmean(all_test_subject_f1s):.4f} +/- {np.nanstd(all_test_subject_f1s):.4f}\n")
        #f.write(f"  Specificity: {np.nanmean(all_test_subject_specificities):.4f} +/- {np.nanstd(all_test_subject_specificities):.4f}\n")

    print(f"\nSummary metrics saved to: {summary_file}")
    
    plots_output_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_output_dir, exist_ok=True)
    classes_for_cm = ['Healthy', 'Schizophrenia'] # Define once

    # 1. Identify the Representative Run (based on subject-level test AUROC closest to mean)
    if run_data_for_plotting and args.num_runs > 0:
        mean_subject_auroc_all_runs = np.nanmean(all_test_subject_aurocs)
        # Sort runs by how close their subject_test_auroc is to the mean
        run_data_for_plotting.sort(key=lambda x: abs(x['subject_test_auroc'] - mean_subject_auroc_all_runs))
        representative_run_data = run_data_for_plotting[0]
        rep_run_idx = representative_run_data['run_idx']
        rep_run_seed = representative_run_data['seed']
        print(f"\nSelected Representative Run: Index {rep_run_idx} (Seed: {rep_run_seed}) for detailed plots.")

        # I. Plot Training and Validation Performance for Representative Run
        plot_training_validation_curves(representative_run_data['epoch_metrics'], plots_output_dir, f"{rep_run_idx}")
    
        if all('segment_accuracy' in r for r in run_data_for_plotting):
            sorted_runs_by_seg_acc = sorted(run_data_for_plotting, key=lambda x: x['segment_accuracy'], reverse=True)
            best_seg_acc_run_data = sorted_runs_by_seg_acc[0] # The first one is the best
        
            best_acc_run_idx = best_seg_acc_run_data['run_idx']
            best_acc_run_seed = best_seg_acc_run_data['seed']
            best_acc_segment_accuracy = best_seg_acc_run_data['segment_accuracy']
            plot_training_validation_curves(
                best_seg_acc_run_data['epoch_metrics'], 
                plots_output_dir, 
                f"BEST_SEG_ACC_run_{best_acc_run_idx}_seed{best_acc_run_seed}" # Updated filename
            )
            print(f"Saved training curves for the run with highest segment-level accuracy to {plots_output_dir}")
        else:
            print("Could not plot for highest segment accuracy run: 'segment_accuracy' key missing in run_data_for_plotting.")

        print(f"\n--- Attempting to plot training curves specifically for Run 1 ---")
        run1_data = None
        for r_data in run_data_for_plotting: # Iterate through the potentially AUROC-sorted list
            if r_data['run_idx'] == 1:
                run1_data = r_data
                break
        
        if run1_data:
            run1_seed = run1_data['seed']
            print(f"Found Run 1 (Seed: {run1_seed}). Plotting its training curves.")
            plot_training_validation_curves(
                run1_data['epoch_metrics'],
                plots_output_dir,
                f"SPECIFIC_run_1_seed{run1_seed}" # Unique identifier for Run 1's plot
            )
            print(f"Saved training curves specifically for Run 1 to {plots_output_dir}")
        else:
            if args.num_runs > 0 : # Only print if we expected Run 1 to exist
                 print(f"Run 1 data not found in run_data_for_plotting. Cannot plot its training curves.")
            # If args.num_runs is 0, it's normal not to find Run 1.


        # III. Detailed Look at Subject-Level Discriminative Ability from Representative Run
        plot_confusion_matrix_custom(
            y_true=representative_run_data['subject_targets'],
            y_pred=representative_run_data['subject_preds'],
            classes=classes_for_cm,
            title=f'Subject-Level Confusion Matrix (Run {rep_run_idx})',
            output_dir=plots_output_dir,
            filename=f'subject_cm_repr_run_{rep_run_idx}.png'
        )
        plot_roc_curve_custom(
            y_true=representative_run_data['subject_targets'],
            y_score=representative_run_data['subject_probs'],
            title=f'Subject-Level ROC Curve (Run {rep_run_idx})',
            output_dir=plots_output_dir,
            filename=f'subject_roc_repr_run_{rep_run_idx}.png'
        )
        plot_precision_recall_curve_custom(
            y_true=representative_run_data['subject_targets'],
            y_score=representative_run_data['subject_probs'],
            title=f'Subject-Level Precision-Recall Curve (Run {rep_run_idx})',
            output_dir=plots_output_dir,
            filename=f'subject_prc_repr_run_{rep_run_idx}.png'
        )
    else:
        print("Not enough run data to select a representative run or generate detailed plots.")


#confusion matrix for the lowest and highest accuracy runs
    if run_data_for_plotting and args.num_runs > 0: # Ensure we have data

        # 1. Segment-Level CM for the Representative Run (selected by subject AUROC)
        if 'representative_run_data' in locals(): # Check if representative_run_data was defined
            plot_confusion_matrix_custom(
                y_true=representative_run_data['segment_targets'],
                y_pred=representative_run_data['segment_preds'],
                classes=classes_for_cm,
                title=f'Segment-Level CM (Rep. Run {representative_run_data["run_idx"]}, Seed {representative_run_data["seed"]})',
                output_dir=plots_output_dir,
                filename=f'segment_cm_repr_run_{representative_run_data["run_idx"]}.png'
            )
        else:
            print("Representative run data not available for segment CM.")

        # 2. Runs with Highest and Lowest Subject-Level Accuracy
        if args.num_runs > 0: # Need at least one run to find min/max
            # Sort by subject_accuracy to find min and max
            # Ensure 'subject_accuracy' was stored in run_data_for_plotting items
            if all('subject_accuracy' in r for r in run_data_for_plotting):
                sorted_by_subject_acc = sorted(run_data_for_plotting, key=lambda x: x['subject_accuracy'])
                
                # Subject-Level CM for LOWEST Subject Accuracy Run
                lowest_subject_acc_run = sorted_by_subject_acc[0]
                plot_confusion_matrix_custom(
                    y_true=lowest_subject_acc_run['subject_targets'],
                    y_pred=lowest_subject_acc_run['subject_preds'],
                    classes=classes_for_cm,
                    title=f'Subject-Level CM (Lowest Subj. Acc: {lowest_subject_acc_run["subject_accuracy"]:.3f} - Run {lowest_subject_acc_run["run_idx"]})',
                    output_dir=plots_output_dir,
                    filename=f'subject_cm_lowest_subj_acc_run_{lowest_subject_acc_run["run_idx"]}.png'
                )

                # Subject-Level CM for HIGHEST Subject Accuracy Run
                highest_subject_acc_run = sorted_by_subject_acc[-1]
                plot_confusion_matrix_custom(
                    y_true=highest_subject_acc_run['subject_targets'],
                    y_pred=highest_subject_acc_run['subject_preds'],
                    classes=classes_for_cm,
                    title=f'Subject-Level CM (Highest Subj. Acc: {highest_subject_acc_run["subject_accuracy"]:.3f} - Run {highest_subject_acc_run["run_idx"]})',
                    output_dir=plots_output_dir,
                    filename=f'subject_cm_highest_subj_acc_run_{highest_subject_acc_run["run_idx"]}.png'
                )
            else:
                print("Subject accuracy not available in all run data for min/max subject CMs.")

        # 3. Runs with Highest and Lowest Segment-Level Accuracy
        if args.num_runs > 0:
            # Sort by segment_accuracy to find min and max
            # Ensure 'segment_accuracy' was stored in run_data_for_plotting items
            if all('segment_accuracy' in r for r in run_data_for_plotting):
                sorted_by_segment_acc = sorted(run_data_for_plotting, key=lambda x: x['segment_accuracy'])

                # Segment-Level CM for LOWEST Segment Accuracy Run
                lowest_segment_acc_run = sorted_by_segment_acc[0]
                plot_confusion_matrix_custom(
                    y_true=lowest_segment_acc_run['segment_targets'],
                    y_pred=lowest_segment_acc_run['segment_preds'],
                    classes=classes_for_cm,
                    title=f'Segment-Level CM (Lowest Seg. Acc: {lowest_segment_acc_run["segment_accuracy"]:.3f} - Run {lowest_segment_acc_run["run_idx"]})',
                    output_dir=plots_output_dir,
                    filename=f'segment_cm_lowest_seg_acc_run_{lowest_segment_acc_run["run_idx"]}.png'
                )

                # Segment-Level CM for HIGHEST Segment Accuracy Run
                highest_segment_acc_run = sorted_by_segment_acc[-1]
                plot_confusion_matrix_custom(
                    y_true=highest_segment_acc_run['segment_targets'],
                    y_pred=highest_segment_acc_run['segment_preds'],
                    classes=classes_for_cm,
                    title=f'Segment-Level CM (Highest Seg. Acc: {highest_segment_acc_run["segment_accuracy"]:.3f} - Run {highest_segment_acc_run["run_idx"]})',
                    output_dir=plots_output_dir,
                    filename=f'segment_cm_highest_seg_acc_run_{highest_segment_acc_run["run_idx"]}.png'
                )
            else:
                print("Segment accuracy not available in all run data for min/max segment CMs.")
                
    else: # This else corresponds to the initial `if run_data_for_plotting and args.num_runs > 0:` for representative run
        if not (run_data_for_plotting and args.num_runs > 0): # If already printed for representative, don't print again
             print("Not enough run data to select representative, min/max accuracy runs or generate detailed plots.")

    print("\nGenerating bar charts for mean performance...")

    # Prepare data for Subject-Level Bar Chart (using metrics you decided on)
    subject_labels = ['Accuracy', 'AUROC', 'Recall (Sensitivity)', 'Precision']
    subject_lists_for_means = [
        all_test_subject_accs,
        all_test_subject_aurocs,
        all_test_subject_recalls,
        all_test_subject_precisions
    ]
    if all(len(l) > 0 for l in subject_lists_for_means):
        subject_means = [np.nanmean(l) for l in subject_lists_for_means]
        subject_stds = [np.nanstd(l) for l in subject_lists_for_means]
        plot_bar_chart_with_errors(
            subject_means,
            subject_stds,
            subject_labels,
            f'Mean Subject-Level Test Performance ({args.num_runs} Runs)',
            plots_output_dir,
            'subject_level_mean_performance_barchart.png'
        )
    else:
        print("Skipping subject-level mean performance bar chart due to missing data in one or more metrics.")

    # Prepare data for Segment-Level Bar Chart (using metrics you decided on)
    segment_labels = ['Accuracy', 'AUROC', 'Recall', 'Precision']
    segment_lists_for_means = [
        all_test_segment_accs,
        all_test_segment_aurocs,
        all_test_segment_recalls,
        all_test_segment_precisions
    ]
    if all(len(l) > 0 for l in segment_lists_for_means):
        segment_means = [np.nanmean(l) for l in segment_lists_for_means]
        segment_stds = [np.nanstd(l) for l in segment_lists_for_means]
        plot_bar_chart_with_errors(
            segment_means,
            segment_stds,
            segment_labels,
            f'Mean Segment-Level Test Performance ({args.num_runs} Runs)',
            plots_output_dir,
            'segment_level_mean_performance_barchart.png'
        )
    else:
        print("Skipping segment-level mean performance bar chart due to missing data in one or more metrics.")


    # II. Visualizing Subject-Level Performance Distribution (Box Plots)
    subject_level_boxplot_data = {
        'Accuracy': all_test_subject_accs,
        'AUROC': all_test_subject_aurocs,
        'Recall (Sensitivity)': all_test_subject_recalls,
        'Precision' : all_test_subject_precisions
        #'Specificity': all_test_subject_specificities,
        #'F1-Score': all_test_subject_f1s
        #'Recall': all_test_subject_recalls,
        
    }
    if all(len(v) > 0 for v in subject_level_boxplot_data.values()): # Check if lists are populated
        plot_boxplots(subject_level_boxplot_data, 'Subject-Level Test', plots_output_dir, 'subject_level_performance')
    else:
        print("Skipping subject-level boxplots as metric lists are empty.")


    # IV. Visualizing Segment-Level Performance Distribution (Box Plots)
    segment_level_boxplot_data = {
        'Accuracy': all_test_segment_accs,
        'AUROC': all_test_segment_aurocs,
        'Recall': all_test_segment_recalls,
        'Precision' : all_test_segment_precisions
        #'F1-Score': all_test_segment_f1s
        # You can add more here if desired, e.g., Accuracy, Specificity
        # 'Specificity': all_test_segment_specificities
      
    }
    if all(len(v) > 0 for v in segment_level_boxplot_data.values()): # Check if lists are populated
         plot_boxplots(segment_level_boxplot_data, 'Segment-Level Test', plots_output_dir, 'segment_level_performance')
    else:
        print("Skipping segment-level boxplots as metric lists are empty.")

    print(f"\nAll plots saved to: {plots_output_dir}")
    
    print("Process completed.")

