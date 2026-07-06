import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings("ignore")

vi = pd.read_csv("vi_train.csv")
target = pd.read_csv("bibc_bcbv_train.csv")
df = vi.merge(target, on=['system_id','topology_id'])

feature_cols = [c for c in vi.columns if c.startswith('V') or c.startswith('I')]
X = df[feature_cols].values.astype(np.float32)
target_cols = [c for c in target.columns if c.startswith('BIBC') or c.startswith('BCBV')]
y = df[target_cols].values.astype(np.float32)

print(f"Training samples: {X.shape[0]}")

x_scaler = StandardScaler()
X = x_scaler.fit_transform(X)

bcbv_scaler = StandardScaler()
y_bcbv = y[:, 289:]
y_bcbv_scaled = bcbv_scaler.fit_transform(y_bcbv)
y_scaled = np.hstack([y[:, :289], y_bcbv_scaled])

X_train, X_val, y_train, y_val = train_test_split(X, y_scaled, test_size=0.2, random_state=42)
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)
y_val_bcbv_original = y_val[:, 289:]

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.drop(out)
        out = self.bn2(self.fc2(out))
        out = self.relu(out + res)
        return self.drop(out)

class Regressor(nn.Module):
    def __init__(self, input_dim=35, hidden=512, output_dim=578, num_blocks=4, dropout=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden, output_dim)
    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.output_layer(x)

model = Regressor()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
criterion_bibc = nn.BCEWithLogitsLoss()
criterion_bcbv = nn.MSELoss()

batch_size = 128
epochs = 300
best_val_loss = float('inf')
dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        out = model(xb)
        bibc_logits = out[:, :289]
        bcbv_pred = out[:, 289:]
        loss = criterion_bibc(bibc_logits, yb[:, :289]) + criterion_bcbv(bcbv_pred, yb[:, 289:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    train_loss = total_loss / len(X_train_t)

    model.eval()
    with torch.no_grad():
        out_val = model(X_val_t)
        bibc_logits_val = out_val[:, :289]
        bcbv_pred_val = out_val[:, 289:]
        bibc_probs = torch.sigmoid(bibc_logits_val)
        bibc_acc = ((bibc_probs > 0.5).float() == y_val_t[:, :289]).float().mean().item()
        bcbv_loss_scaled = criterion_bcbv(bcbv_pred_val, y_val_t[:, 289:]).item()
        bcbv_pred_orig = bcbv_scaler.inverse_transform(bcbv_pred_val.cpu().numpy())
        bcbv_true_orig = y_val_bcbv_original
        rmse_overall = np.sqrt(np.mean((bcbv_pred_orig - bcbv_true_orig)**2))
        active = bcbv_true_orig > 1e-6
        rmse_active = np.sqrt(np.mean((bcbv_pred_orig[active] - bcbv_true_orig[active])**2)) if np.any(active) else np.nan

    val_loss_sum = criterion_bibc(bibc_logits_val, y_val_t[:, :289]).item() + bcbv_loss_scaled
    scheduler.step(val_loss_sum)
    if val_loss_sum < best_val_loss:
        best_val_loss = val_loss_sum
        torch.save(model.state_dict(), "unified_best.pth")
        print(f"Epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss_sum:.4f} | "
              f"BIBC acc {bibc_acc:.4f} | BCBV RMSE overall {rmse_overall:.6f} | active {rmse_active:.6f} (best)")
    elif epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss_sum:.4f} | "
              f"BIBC acc {bibc_acc:.4f} | BCBV RMSE overall {rmse_overall:.6f} | active {rmse_active:.6f}")

torch.save(model.state_dict(), "unified_final.pth")
joblib.dump(x_scaler, "x_scaler.pkl")
joblib.dump(bcbv_scaler, "bcbv_scaler.pkl")
print(f"Training done. Best val loss: {best_val_loss:.4f}")