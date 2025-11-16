import numpy as np
import os
from scipy import signal
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import mne
# from mne.preprocessing import ICA
import warnings

# (The _apply_ica_cleaning function, if used, would remain unchanged)
# ...

def load_schizophrenia_data(healthy_dir, schizophrenia_dir):
    """
    Load the schizophrenia EEG dataset from the specified directories.
    (This function remains unchanged from your previous version)
    """
    all_data = []
    labels = []
    for filename in os.listdir(healthy_dir):
        if filename.endswith('.eea'): # Assuming .eea as per original
            file_path = os.path.join(healthy_dir, filename)
            raw_data = np.loadtxt(file_path)
            subject_data = raw_data.reshape(16, 7680)
            all_data.append(subject_data)
            labels.append(0)
    for filename in os.listdir(schizophrenia_dir):
        if filename.endswith('.eea'): # Assuming .eea
            file_path = os.path.join(schizophrenia_dir, filename)
            raw_data = np.loadtxt(file_path)
            subject_data = raw_data.reshape(16, 7680)
            all_data.append(subject_data)
            labels.append(1)
    X = np.array(all_data)
    y = np.array(labels)
    return X, y

def preprocess_schizophrenia_data(X, segment_duration_seconds=12, sampling_rate=128):
    """
    Preprocess the schizophrenia EEG data.
    Input X shape: (n_subjects, n_channels, n_samples_raw)
    Output X_final shape: (n_total_segments, n_freq, n_time, n_channels)
    (This function remains unchanged from your previous version)
    """
    n_subjects, n_channels, n_samples_raw = X.shape
    segment_samples = segment_duration_seconds * sampling_rate
    n_segments_per_subject = n_samples_raw // segment_samples

    X_segmented = np.zeros((n_subjects * n_segments_per_subject, n_channels, segment_samples))
    for subject_idx in range(n_subjects):
        for segment_idx in range(n_segments_per_subject):
            start_idx = segment_idx * segment_samples
            end_idx = start_idx + segment_samples
            X_segmented[subject_idx * n_segments_per_subject + segment_idx] = X[subject_idx, :, start_idx:end_idx]

    b, a = signal.butter(4, [0.5, 45], btype='bandpass', fs=sampling_rate)
    X_filtered = np.zeros_like(X_segmented)
    for i in range(X_segmented.shape[0]):
        for j in range(n_channels):
            X_filtered[i, j] = signal.filtfilt(b, a, X_segmented[i, j])

    nperseg = 256
    noverlap = 128
    
    # Determine STFT output shape from one example
    _freqs, _times, Zxx_example = signal.stft(X_filtered[0, 0, :], fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
    n_freqs = Zxx_example.shape[0]
    n_times = Zxx_example.shape[1]

    X_stft = np.zeros((X_filtered.shape[0], n_channels, n_freqs, n_times), dtype=np.complex64)
    for i in range(X_filtered.shape[0]): # Iterate over segments
        for j in range(n_channels): # Iterate over channels
            _freqs, _times, Zxx = signal.stft(X_filtered[i, j, :], fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
            X_stft[i, j, :, :] = Zxx
    
    X_stft_mag = np.abs(X_stft)
    # Transpose to: (n_segments, n_freq, n_time, n_channels)
    # This format is suitable for channel-wise operations later
    X_final = np.transpose(X_stft_mag, (0, 2, 3, 1))
    return X_final


# --- MODIFIED PCA FUNCTION FOR CHANNEL-WISE PCA ---
def apply_channel_wise_pca(X_train_processed, X_val_processed, X_test_processed, n_components_per_channel):
    """
    Applies PCA channel-wise to the processed EEG data.
    Input X_processed shape: (n_segments, n_freq, n_time, n_channels)

    Args:
        X_train_processed (np.ndarray): Processed training data.
        X_val_processed (np.ndarray): Processed validation data.
        X_test_processed (np.ndarray): Processed test data.
        n_components_per_channel: PCA n_components parameter for each channel.

    Returns:
        tuple: (X_train_pca_concat, X_val_pca_concat, X_test_pca_concat, list_of_pca_models)
    """
    n_segments_train, n_freq, n_time, n_channels = X_train_processed.shape
    n_segments_val = X_val_processed.shape[0]
    n_segments_test = X_test_processed.shape[0]

    print(f"\n--- Applying Channel-Wise PCA ---")
    print(f"  Input data shape (train): ({n_segments_train}, {n_freq}, {n_time}, {n_channels})")
    print(f"  PCA n_components target per channel: {n_components_per_channel}")

    X_train_pca_channels = []
    X_val_pca_channels = []
    X_test_pca_channels = []
    pca_models_per_channel = []
    scalers_per_channel = []
    
    total_original_features_per_segment = 0
    total_pca_features_per_segment = 0

    for ch_idx in range(n_channels):
        # Extract data for the current channel: (n_segments, n_freq, n_time)
        train_ch_data = X_train_processed[:, :, :, ch_idx]
        val_ch_data = X_val_processed[:, :, :, ch_idx]
        test_ch_data = X_test_processed[:, :, :, ch_idx]

        # Flatten features for this channel: (n_segments, n_freq * n_time)
        train_ch_flat = train_ch_data.reshape(n_segments_train, -1)
        val_ch_flat = val_ch_data.reshape(n_segments_val, -1)
        test_ch_flat = test_ch_data.reshape(n_segments_test, -1)
        
        #Now train_ch_flat has shape (n_segments, n_freq * n_time)
        original_features_this_channel = train_ch_flat.shape[1]
        total_original_features_per_segment += original_features_this_channel

        # Scale data for this channel
        scaler = StandardScaler()
        #the following steps standardize(normalise) the data to prevent any feature from dominating the pca
        #fit computes the mean and standard deviation for each feature and trasform standardizes the training data using this
        train_ch_scaled = scaler.fit_transform(train_ch_flat)
        val_ch_scaled = scaler.transform(val_ch_flat)
        test_ch_scaled = scaler.transform(test_ch_flat)
        scalers_per_channel.append(scaler)

        # Apply PCA for this channel
        pca = PCA(n_components=n_components_per_channel, random_state=42)
        train_ch_pca = pca.fit_transform(train_ch_scaled)
        val_ch_pca = pca.transform(val_ch_scaled)
        test_ch_pca = pca.transform(test_ch_scaled)
        
        pca_features_this_channel = train_ch_pca.shape[1]
        total_pca_features_per_segment += pca_features_this_channel
        # print(f"  Channel {ch_idx+1}: Original features = {original_features_this_channel}, PCA features = {pca_features_this_channel}, Explained variance = {np.sum(pca.explained_variance_ratio_):.4f}")


        X_train_pca_channels.append(train_ch_pca)
        X_val_pca_channels.append(val_ch_pca)
        X_test_pca_channels.append(test_ch_pca)
        pca_models_per_channel.append(pca)

    # Concatenate PCA outputs from all channels
    X_train_pca_concat = np.concatenate(X_train_pca_channels, axis=1)
    X_val_pca_concat = np.concatenate(X_val_pca_channels, axis=1)
    X_test_pca_concat = np.concatenate(X_test_pca_channels, axis=1)
    
    print(f"  Total original features per segment (sum over channels): {total_original_features_per_segment}")
    print(f"  Total features after channel-wise PCA (concatenated): {X_train_pca_concat.shape[1]}")

    return X_train_pca_concat, X_val_pca_concat, X_test_pca_concat, pca_models_per_channel

# --- END OF MODIFIED PCA FUNCTION ---

def create_schizophrenia_datasets(healthy_dir, schizophrenia_dir, test_size, validation_ratio, batch_size, random_state, apply_pca_flag=True, pca_n_components=0.95):
    X_raw, y = load_schizophrenia_data(healthy_dir, schizophrenia_dir)

    X_train_raw, X_test_raw, y_train_subjects, y_test_subjects = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train_sub_raw, X_val_raw, y_train_sub_subjects, y_val_subjects = train_test_split(
        X_train_raw, y_train_subjects, test_size=validation_ratio, random_state=random_state, stratify=y_train_subjects
    )

    # Preprocess data: Output shape (n_segments, n_freq, n_time, n_channels)
    X_train_processed = preprocess_schizophrenia_data(X_train_sub_raw)
    X_val_processed = preprocess_schizophrenia_data(X_val_raw)
    X_test_processed = preprocess_schizophrenia_data(X_test_raw)

    n_segments_per_subject = X_train_processed.shape[0] // len(y_train_sub_subjects) if len(y_train_sub_subjects) > 0 else 0
    if n_segments_per_subject == 0 and X_train_processed.shape[0] > 0:
        n_segments_per_subject = 1 

    y_train_expanded = np.repeat(y_train_sub_subjects, n_segments_per_subject)
    y_val_expanded = np.repeat(y_val_subjects, n_segments_per_subject)
    y_test_expanded = np.repeat(y_test_subjects, n_segments_per_subject)

    
    # Apply channel-wise PCA using the processed data directly
    X_train_final, X_val_final, X_test_final, _ = apply_channel_wise_pca(
        X_train_processed, X_val_processed, X_test_processed, n_components_per_channel=pca_n_components
    )
    

    X_train_tensor = torch.FloatTensor(X_train_final)
    y_train_tensor = torch.tensor(y_train_expanded, dtype=torch.int64)
    X_val_tensor = torch.FloatTensor(X_val_final)
    y_val_tensor = torch.tensor(y_val_expanded, dtype=torch.int64)
    X_test_tensor = torch.FloatTensor(X_test_final)
    y_test_tensor = torch.tensor(y_test_expanded, dtype=torch.int64)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nDataset statistics:")
    print(f"  Total subjects: {len(y)}")
    print(f"  Training subjects: {len(y_train_sub_subjects)}")
    print(f"  Validation subjects: {len(y_val_subjects)}")
    print(f"  Test subjects: {len(y_test_subjects)} ({100 * len(y_test_subjects) / len(y) if len(y) > 0 else 0:.1f}%)")
    print(f"  Segments per subject: {n_segments_per_subject}")
    print(f"  Total training segments: {len(y_train_expanded)}")
    print(f"  Total validation segments: {len(y_val_expanded)}")
    print(f"  Total test segments: {len(y_test_expanded)}")
    print(f"  Number of features for model: {X_train_final.shape[1]}")

    return train_loader, val_loader, test_loader, X_train_final.shape[1], 2, len(y_test_subjects), n_segments_per_subject