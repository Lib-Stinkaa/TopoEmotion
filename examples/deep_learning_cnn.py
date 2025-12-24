import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 10


class MultimodalDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MultimodalCNN(nn.Module):
    def __init__(self, n_signals=8, n_features_per_signal=518, num_classes=2):
        super(MultimodalCNN, self).__init__()

        self.n_signals = n_signals
        self.n_features_per_signal = n_features_per_signal

        self.signal_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=5, padding=2),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.3),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(32),
                nn.Dropout(0.3),
            )
            for _ in range(n_signals)
        ])

        fused_dim = n_signals * 32 * 32

        self.fusion_layers = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.n_signals, self.n_features_per_signal)

        signal_features = []
        for i in range(self.n_signals):
            signal_x = x[:, i:i+1, :]
            signal_feat = self.signal_convs[i](signal_x)
            signal_features.append(signal_feat)

        fused = torch.cat(signal_features, dim=1)
        fused = fused.view(batch_size, -1)

        out = self.fusion_layers(fused)
        return out


def prepare_data(df):
    feature_cols = [col for col in df.columns if col.startswith(('H0_', 'H1_', 'landscape_'))]
    meta_cols = ['subject_id', 'video_id', 'start_time', 'valence', 'arousal']

    df['window_key'] = (df['subject_id'].astype(str) + '_' +
                        df['video_id'].astype(str) + '_' +
                        df['start_time'].astype(str))

    unique_windows = df[meta_cols + ['window_key']].drop_duplicates('window_key')
    signal_types = sorted(df['signal_type'].unique())

    multimodal_features = []
    for signal in signal_types:
        signal_df = df[df['signal_type'] == signal][['window_key'] + feature_cols].copy()
        signal_df.columns = ['window_key'] + [f'{signal}_{col}' for col in feature_cols]
        multimodal_features.append(signal_df)

    merged_df = unique_windows.copy()
    for signal_df in multimodal_features:
        merged_df = merged_df.merge(signal_df, on='window_key', how='left')

    merged_df = merged_df.drop('window_key', axis=1).dropna()

    fusion_cols = [col for col in merged_df.columns
                   if any(col.startswith(f'{sig}_') for sig in signal_types)]

    return merged_df, fusion_cols


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    preds, labels = [], []

    for features, targets in loader:
        features, targets = features.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds.extend(outputs.argmax(dim=1).cpu().numpy())
        labels.extend(targets.cpu().numpy())

    return total_loss / len(loader), accuracy_score(labels, preds)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    preds, labels = [], []

    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            outputs = model(features)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            preds.extend(outputs.argmax(dim=1).cpu().numpy())
            labels.extend(targets.cpu().numpy())

    return {
        'loss': total_loss / len(loader),
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted', zero_division=0),
        'predictions': np.array(preds),
        'labels': np.array(labels)
    }


def loso_cross_validation(df, feature_cols, target='valence'):
    subjects = sorted(df['subject_id'].unique())
    all_results = []

    for test_subject in subjects:
        train_df = df[df['subject_id'] != test_subject].copy()
        test_df = df[df['subject_id'] == test_subject].copy()

        target_col = f'{target}_binary'
        train_df[target_col] = (train_df[target] > 5.0).astype(int)
        test_df[target_col] = (test_df[target] > 5.0).astype(int)

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        val_size = int(len(X_train_scaled) * 0.2)
        X_val = X_train_scaled[-val_size:]
        y_val = y_train[-val_size:]
        X_train_only = X_train_scaled[:-val_size]
        y_train_only = y_train[:-val_size]

        train_loader = DataLoader(MultimodalDataset(X_train_only, y_train_only),
                                  batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(MultimodalDataset(X_val, y_val),
                               batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(MultimodalDataset(X_test_scaled, y_test),
                                batch_size=BATCH_SIZE, shuffle=False)

        model = MultimodalCNN(
            n_signals=8,
            n_features_per_signal=len(feature_cols) // 8,
            num_classes=2
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_val_f1 = 0
        patience_counter = 0
        best_model_state = None

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
            val_results = evaluate(model, val_loader, criterion)

            if val_results['f1'] > best_val_f1:
                best_val_f1 = val_results['f1']
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        test_results = evaluate(model, test_loader, criterion)
        all_results.append({
            'subject_id': test_subject,
            'accuracy': test_results['accuracy'],
            'f1': test_results['f1']
        })

    results_df = pd.DataFrame(all_results)
    return {
        'accuracy_mean': results_df['accuracy'].mean(),
        'accuracy_std': results_df['accuracy'].std(),
        'f1_mean': results_df['f1'].mean(),
        'f1_std': results_df['f1'].std()
    }


def main(features_file):
    df = pd.read_csv(features_file)
    multimodal_df, fusion_cols = prepare_data(df)

    valence = loso_cross_validation(multimodal_df, fusion_cols, target='valence')
    arousal = loso_cross_validation(multimodal_df, fusion_cols, target='arousal')

    print(f"CNN - Valence: Acc={valence['accuracy_mean']:.4f}±{valence['accuracy_std']:.4f}, "
          f"F1={valence['f1_mean']:.4f}±{valence['f1_std']:.4f}")
    print(f"CNN - Arousal: Acc={arousal['accuracy_mean']:.4f}±{arousal['accuracy_std']:.4f}, "
          f"F1={arousal['f1_mean']:.4f}±{arousal['f1_std']:.4f}")


if __name__ == "__main__":
    import sys
    features_file = sys.argv[1] if len(sys.argv) > 1 else 'all_features.csv'
    main(features_file)
