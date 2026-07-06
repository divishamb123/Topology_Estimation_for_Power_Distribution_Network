import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import argparse
import sys

np.set_printoptions(threshold=np.inf, suppress=True, precision=4, linewidth=200)

# -------------------------------------------------------------------
# 1. MODEL DEFINITION
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -------------------------------------------------------------------
def load_model_and_scalers():
    model = Regressor()
    model.load_state_dict(torch.load("unified_final.pth", map_location="cpu"))
    model.eval()
    x_scaler = joblib.load("x_scaler.pkl")
    bcbv_scaler = joblib.load("bcbv_scaler.pkl")
    return model, x_scaler, bcbv_scaler

def get_dims(system_id):
    system_id = str(system_id).replace('.0', '')
    dims = {'5': (5,4), '15': (15,14), '18': (18,17)}
    if system_id not in dims:
        raise ValueError(f"Unknown system ID: {system_id}. Expected '5', '15', or '18'.")
    return dims[system_id]

def enforce_bcbv_consistency(bcbv_mat, active_mask, method='median'):
    bcbv_mat = bcbv_mat.copy()
    num_non_slack, num_branches = bcbv_mat.shape
    for j in range(num_branches):
        col_active = active_mask[:, j] > 0
        if not np.any(col_active):
            continue
        active_vals = bcbv_mat[col_active, j]
        if method == 'median':
            consensus = np.median(active_vals)
        elif method == 'mean':
            consensus = np.mean(active_vals)
        elif method == 'diagonal':
            if j < num_non_slack and active_mask[j, j] > 0:
                consensus = bcbv_mat[j, j]
            else:
                consensus = np.median(active_vals)
        else:
            raise ValueError(f"Unknown method: {method}")
        bcbv_mat[col_active, j] = consensus
    return bcbv_mat

def predict_sample(model, x_scaler, bcbv_scaler, vals, system_id, bcbv_method='median'):
    num_buses, num_branches = get_dims(system_id)
    num_non_slack = num_buses - 1
    if len(vals) < 35:
        vals = list(vals) + [0.0] * (35 - len(vals))
        
    X = np.array(vals[:35]).reshape(1, -1)
    X_scaled = x_scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        out = model(X_t)
        bibc_logits = out[0, :289]
        bcbv_scaled = out[0, 289:]
        bibc_probs = torch.sigmoid(bibc_logits)
        bibc_pred = (bibc_probs > 0.5).int().numpy()
        bcbv_pred = bcbv_scaler.inverse_transform(bcbv_scaled.numpy().reshape(1, -1)).flatten()

    bibc_mat = bibc_pred[:num_branches * num_non_slack].reshape(num_branches, num_non_slack)
    bcbv_mat = bcbv_pred[:num_non_slack * num_branches].reshape(num_non_slack, num_branches)

    active_mask = bibc_mat.T
    bcbv_mat = bcbv_mat * active_mask
    bcbv_mat = np.where(np.abs(bcbv_mat) < 1e-10, 0.0, bcbv_mat)
    bcbv_mat = enforce_bcbv_consistency(bcbv_mat, active_mask, method=bcbv_method)
    return bibc_mat, bcbv_mat

def prepare_true_bcbv(true_bibc, true_bcbv_raw, bcbv_method='median'):
    true_mask = true_bibc.T
    bcbv_masked = true_bcbv_raw * true_mask
    bcbv_masked = np.where(np.abs(bcbv_masked) < 1e-10, 0.0, bcbv_masked)
    bcbv_masked = enforce_bcbv_consistency(bcbv_masked, true_mask, method=bcbv_method)
    return bcbv_masked

# -------------------------------------------------------------------
# 3. EVALUATION LOGIC
# -------------------------------------------------------------------
def print_comparison(true_bibc, pred_bibc, true_bcbv_raw, pred_bcbv, num_branches, num_non_slack, bcbv_method='median'):
    true_bcbv = prepare_true_bcbv(true_bibc, true_bcbv_raw, bcbv_method)
    acc = np.sum(true_bibc == pred_bibc)
    total = num_branches * num_non_slack
    print(f"Accuracy: {acc}/{total} correct ({acc/total*100:.1f}%)")
    
    print("\n[ TRUE BIBC ]")
    print(true_bibc.astype(int))
    print("\n[ PRED BIBC ]")
    print(pred_bibc.astype(int))

    true_mask = true_bibc.T
    active = true_mask > 0.5
    if np.any(active):
        mae = np.mean(np.abs(pred_bcbv[active] - true_bcbv[active]))
        print(f"\nMean Absolute Error (on active branches): {mae:.6f} ohms")
    else:
        print("\nNo active branches in true BCBV.")

    true_bcbv_rounded = np.where(np.abs(true_bcbv) < 1e-4, 0, true_bcbv)
    pred_bcbv_rounded = np.where(np.abs(pred_bcbv) < 1e-4, 0, pred_bcbv)
    print("\n[ TRUE BCBV ]")
    print(np.round(true_bcbv_rounded, 4))
    print("\n[ PREDICTED BCBV ]")
    print(np.round(pred_bcbv_rounded, 4))

def evaluate_entire_dataset(df, vi_cols, target_cols, model, x_scaler, bcbv_scaler, bcbv_method):
    print(f"\nEvaluating entire dataset ({len(df)} samples)... This might take a moment.")
    total_bibc_correct = 0
    total_bibc_elements = 0
    bcbv_maes = []

    for i in range(len(df)):
        row = df.iloc[i]
        system_id = str(row['system_id']).replace('.0', '')
        
        vals = row[vi_cols].values.astype(float)
        true_bibc_flat = row[target_cols[:289]].values.astype(float)
        true_bcbv_flat = row[target_cols[289:]].values.astype(float)
        
        num_buses, num_branches = get_dims(system_id)
        num_non_slack = num_buses - 1
        
        true_bibc = true_bibc_flat[:num_branches * num_non_slack].reshape(num_branches, num_non_slack)
        true_bcbv_raw = true_bcbv_flat[:num_non_slack * num_branches].reshape(num_non_slack, num_branches)
        
        pred_bibc, pred_bcbv = predict_sample(model, x_scaler, bcbv_scaler, vals, system_id, bcbv_method)
        true_bcbv = prepare_true_bcbv(true_bibc, true_bcbv_raw, bcbv_method)

        # Metrics
        total_bibc_correct += np.sum(true_bibc == pred_bibc)
        total_bibc_elements += (num_branches * num_non_slack)
        
        true_mask = true_bibc.T
        active = true_mask > 0.5
        if np.any(active):
            mae = np.mean(np.abs(pred_bcbv[active] - true_bcbv[active]))
            bcbv_maes.append(mae)
            
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(df)} samples...")

    final_acc = (total_bibc_correct / total_bibc_elements) * 100
    avg_mae = np.mean(bcbv_maes) if bcbv_maes else 0.0

    print("\n" + "="*50)
    print("FINAL TRAINING DATASET PERFORMANCE")
    print("="*50)
    print(f"Total BIBC Accuracy: {final_acc:.2f}% ({total_bibc_correct}/{total_bibc_elements} cells)")
    print(f"Average BCBV MAE:    {avg_mae:.6f} ohms (on active branches)")
    print("="*50)

# -------------------------------------------------------------------
# 4. MAIN
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate BIBC and BCBV on the Training dataset.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Check a specific sample index (0-based). Leave empty to evaluate the whole dataset.")
    parser.add_argument("--vi-train", default="vi_train.csv", help="Training V/I CSV file")
    parser.add_argument("--target-train", default="bibc_bcbv_train.csv", help="Training target CSV file")
    parser.add_argument("--bcbv-method", default="median", choices=["median", "mean", "diagonal"],
                        help="How to reconcile a BCBV column's active entries into one consensus value")
    args = parser.parse_args()

    model, x_scaler, bcbv_scaler = load_model_and_scalers()

    try:
        vi_train = pd.read_csv(args.vi_train, dtype={'system_id': str})
        target_train = pd.read_csv(args.target_train, dtype={'system_id': str})
        df = vi_train.merge(target_train, on=['system_id', 'topology_id'])
        
        if len(df) == 0:
            print("Error: The merged training dataframe is empty. Check your CSVs and column names.")
            sys.exit(1)
            
        vi_cols = [c for c in vi_train.columns if c.startswith('V') or c.startswith('I')]
        target_cols = [c for c in target_train.columns if c.startswith('BIBC') or c.startswith('BCBV')]

        if args.sample is not None:
            # Check a single sample
            if args.sample >= len(df) or args.sample < 0:
                print(f"Sample index {args.sample} out of range (0..{len(df)-1})")
                sys.exit(1)
                
            row = df.iloc[args.sample]
            system_id = str(row['system_id']).replace('.0', '')
            topo_id = row['topology_id']
            vals = row[vi_cols].values.astype(float)
            true_bibc_flat = row[target_cols[:289]].values.astype(float)
            true_bcbv_flat = row[target_cols[289:]].values.astype(float)
            
            num_buses, num_branches = get_dims(system_id)
            num_non_slack = num_buses - 1
            
            true_bibc = true_bibc_flat[:num_branches * num_non_slack].reshape(num_branches, num_non_slack)
            true_bcbv_raw = true_bcbv_flat[:num_non_slack * num_branches].reshape(num_non_slack, num_branches)
            
            pred_bibc, pred_bcbv = predict_sample(model, x_scaler, bcbv_scaler, vals, system_id, args.bcbv_method)

            print(f"\n--- TRAINING DATA: SAMPLE {args.sample} ---")
            print(f"System: {system_id}-bus")
            print(f"Topology ID: {topo_id}")
            print(f"\n--- MATRIX DIMS ({num_branches}x{num_non_slack}) ---")
            print_comparison(true_bibc, pred_bibc, true_bcbv_raw, pred_bcbv, num_branches, num_non_slack, args.bcbv_method)
            
        else:
            # Evaluate the whole dataset
            evaluate_entire_dataset(df, vi_cols, target_cols, model, x_scaler, bcbv_scaler, args.bcbv_method)

    except FileNotFoundError:
        print(f"Could not find {args.vi_train} or {args.target_train}. Make sure the files are in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()