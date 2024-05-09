import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_data_from_folders(base_dir, activity_folders):
    column_names = ['time', 'avg_rss12', 'var_rss12', 'avg_rss13', 'var_rss13', 'avg_rss23', 'var_rss23']
    all_data = pd.DataFrame()
    label_encoder = LabelEncoder()
    scaler = StandardScaler()
    
    # Load data from each folder
    for activity in activity_folders:
        folder_path = os.path.join(base_dir, activity)
        files = [f for f in os.listdir(folder_path) if not os.path.isdir(os.path.join(folder_path, f))]
        activity_data = pd.DataFrame()
        
        for file in files:
            file_path = os.path.join(folder_path, file)
            try:
                # Ensure data does not include any non-numeric values or unexpected strings
                temp_df = pd.read_csv(file_path, comment='#', header=None, names=column_names, on_bad_lines='skip')
                temp_df = temp_df.dropna()  # Drop any rows with NaN values
                temp_df = temp_df[~temp_df.applymap(lambda x: isinstance(x, str)).any(axis=1)]  # Remove any rows that contain strings
                temp_df['label'] = activity
                activity_data = pd.concat([activity_data, temp_df], ignore_index=True)
            except pd.errors.ParserError as e:
                print(f"Skipping file {file} due to errors: {e}")
        
        all_data = pd.concat([all_data, activity_data], ignore_index=True)

    # Encode labels
    all_data['label'] = label_encoder.fit_transform(all_data['label'])
    
    # Check for non-numeric values
    feature_columns = ['avg_rss12', 'var_rss12', 'avg_rss13', 'var_rss13', 'avg_rss23', 'var_rss23']
    all_data[feature_columns] = all_data[feature_columns].apply(pd.to_numeric, errors='coerce')  # Coerce any remaining non-numeric entries to NaN, and then drop them
    all_data = all_data.dropna(subset=feature_columns)
    
    # Normalize features
    all_data[feature_columns] = scaler.fit_transform(all_data[feature_columns])
    
    return all_data, label_encoder.classes_




# Create sequences from the data
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    feature_columns = ['avg_rss12', 'var_rss12', 'avg_rss13', 'var_rss13', 'avg_rss23', 'var_rss23']
    for i in range(len(data) - seq_length):
        seq = data[feature_columns].iloc[i:i + seq_length].values
        label = data['label'].iloc[i + seq_length - 1]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Parameters
base_dir = '/Users/habib/Desktop/DL/data/Activity-Recognition-system-based-on-Multisensor-data-fusion-(AReM)'
activity_folders = ['bending1', 'bending2', 'cycling', 'lying', 'sitting', 'standing', 'walking']
seq_length = 50

# Load and preprocess data
data, classes = load_data_from_folders(base_dir, activity_folders)

# Generate sequences
sequences, labels = create_sequences(data, seq_length)

# Split data and convert to tensors
X_train_val, X_test, y_train_val, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 20% of the entire dataset for validation

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

print(train_loader)
