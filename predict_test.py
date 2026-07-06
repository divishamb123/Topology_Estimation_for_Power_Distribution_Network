import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import argparse
import sys

np.set_printoptions(threshold=np.inf, suppress=True, precision=4, linewidth=200)

# -------------------------------------------------------------------
# 1. MODEL DEFINITION (must match training)
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
    """
    Predict BIBC and BCBV. Mask BCBV with predicted BIBC (transposed)
    and enforce column-wise consistency.
    """
    num_buses, num_branches = get_dims(system_id)
    num_non_slack = num_buses - 1
    if len(vals) < 35:
        vals = vals + [0.0] * (35 - len(vals))
        
    X = np.array(vals[:35]).reshape(1, -1)
    X_scaled = x_scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        out = model(X_t)
        bibc_logits = out[0, :289]
        bcbv_scaled = out[0, 289:]
        bibc_probs = torch.sigmoid(bibc_logits)
        bibc_pred = (bibc_probs > 0.5).int().numpy()
        # Direct inverse transform (removed the /1000 division that zeroed out values)
        bcbv_pred = bcbv_scaler.inverse_transform(bcbv_scaled.numpy().reshape(1, -1)).flatten()

    bibc_mat = bibc_pred[:num_branches * num_non_slack].reshape(num_branches, num_non_slack)
    bcbv_mat = bcbv_pred[:num_non_slack * num_branches].reshape(num_non_slack, num_branches)

    # Mask with predicted BIBC (transpose) and enforce consistency
    active_mask = bibc_mat.T
    bcbv_mat = bcbv_mat * active_mask
    bcbv_mat = np.where(np.abs(bcbv_mat) < 1e-10, 0.0, bcbv_mat)
    bcbv_mat = enforce_bcbv_consistency(bcbv_mat, active_mask, method=bcbv_method)
    return bibc_mat, bcbv_mat

def prepare_true_bcbv(true_bibc, true_bcbv_raw, bcbv_method='median'):
    """Mask true BCBV with true BIBC and enforce consistency."""
    true_mask = true_bibc.T
    bcbv_masked = true_bcbv_raw * true_mask
    bcbv_masked = np.where(np.abs(bcbv_masked) < 1e-10, 0.0, bcbv_masked)
    bcbv_masked = enforce_bcbv_consistency(bcbv_masked, true_mask, method=bcbv_method)
    return bcbv_masked

def print_comparison(true_bibc, pred_bibc, true_bcbv_raw, pred_bcbv, num_branches, num_non_slack, bcbv_method='median'):
    """
    Print BIBC accuracy, true/pred BIBC, MAE on active branches, and
    true/pred BCBV (both masked and enforced).
    """
    # Prepare true BCBV (mask with true BIBC and enforce)
    true_bcbv = prepare_true_bcbv(true_bibc, true_bcbv_raw, bcbv_method)

    # BIBC accuracy
    acc = np.sum(true_bibc == pred_bibc)
    total = num_branches * num_non_slack
    print(f"Accuracy: {acc}/{total} correct ({acc/total*100:.1f}%)")
    
    print("\n[ TRUE BIBC ]")
    print(true_bibc.astype(int))
    
    print("\n[ PRED BIBC ]")
    print(pred_bibc.astype(int))

    # BCBV MAE on true active branches
    true_mask = true_bibc.T
    active = true_mask > 0.5
    if np.any(active):
        mae = np.mean(np.abs(pred_bcbv[active] - true_bcbv[active]))
        print(f"\nMean Absolute Error (on active branches): {mae:.6f} ohms")
    else:
        print("\nNo active branches in true BCBV.")

    # Round for clean output
    true_bcbv_rounded = np.where(np.abs(true_bcbv) < 1e-4, 0, true_bcbv)
    pred_bcbv_rounded = np.where(np.abs(pred_bcbv) < 1e-4, 0, pred_bcbv)
    
    print("\n[ TRUE BCBV ]")
    print(np.round(true_bcbv_rounded, 4))
    
    print("\n[ PREDICTED BCBV ]")
    print(np.round(pred_bcbv_rounded, 4))

def print_predicted_only(pred_bibc, pred_bcbv):
    print("\n--- PREDICTED BIBC MATRIX ---")
    print(pred_bibc.astype(int))
    pred_bcbv_rounded = np.where(np.abs(pred_bcbv) < 1e-4, 0, pred_bcbv)
    print("\n--- PREDICTED BCBV MATRIX (ohms) ---")
    print(np.round(pred_bcbv_rounded, 4))

def read_multi_line_input(prompt):
    print(prompt)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().lower() in ('q', 'done'):
            return None
        if line.strip() == '':
            if lines:
                break
            else:
                continue
        lines.append(line)
    return ' '.join(lines)

def detect_system_from_values(vals):
    voltage_vals = vals[:18]
    nz_voltage = sum(1 for v in voltage_vals if abs(v) > 1e-6)
    if nz_voltage == 5:
        return '5'
    elif nz_voltage == 15:
        return '15'
    elif nz_voltage == 18:
        return '18'
    else:
        nz_total = sum(1 for v in vals if abs(v) > 1e-6)
        if nz_total == 9:
            return '5'
        elif nz_total == 29:
            return '15'
        elif nz_total == 35:
            return '18'
        else:
            if len(vals) == 9:
                return '5'
            elif len(vals) == 29:
                return '15'
            elif len(vals) == 35:
                return '18'
            else:
                raise ValueError(f"Cannot detect system from {len(vals)} values (nz_voltage={nz_voltage})")

# -------------------------------------------------------------------
# 3. GROUND-TRUTH LOOKUP (uses test CSVs)
# -------------------------------------------------------------------
_test_data_cache = {}

def load_test_dataframes(vi_path, target_path):
    key = (vi_path, target_path)
    if key in _test_data_cache:
        return _test_data_cache[key]
    try:
        vi_test = pd.read_csv(vi_path, dtype={'system_id': str})
        target_test = pd.read_csv(target_path, dtype={'system_id': str})
        df = vi_test.merge(target_test, on=['system_id', 'topology_id'])
        df['system_id'] = df['system_id'].astype(str).str.replace('.0', '', regex=False)
        _test_data_cache[key] = (df, vi_test.columns, target_test.columns)
    except Exception:
        _test_data_cache[key] = None
    return _test_data_cache[key]

def find_ground_truth(vals, system_id, df_info, tol=1e-4):
    if df_info is None:
        return None
    df, vi_cols, target_cols = df_info
    feature_cols = [c for c in vi_cols if c.startswith('V') or c.startswith('I')]
    target_cols = [c for c in target_cols if c.startswith('BIBC') or c.startswith('BCBV')]
    n_feat = len(feature_cols)
    padded = list(vals) + [0.0] * max(0, n_feat - len(vals))
    padded = np.array(padded[:n_feat], dtype=float)

    sub = df[df['system_id'] == str(system_id)]
    if len(sub) == 0:
        return None

    feat_matrix = sub[feature_cols].values.astype(float)
    diffs = np.max(np.abs(feat_matrix - padded), axis=1)
    best_idx = int(np.argmin(diffs))
    if diffs[best_idx] > tol:
        return None

    row = sub.iloc[best_idx]
    num_buses, num_branches = get_dims(system_id)
    num_non_slack = num_buses - 1
    # The CSV stores flattened BIBC (branch x non-slack) then BCBV (non-slack x branch)
    true_bibc_flat = row[target_cols[:289]].values.astype(float)
    true_bcbv_flat = row[target_cols[289:]].values.astype(float)
    true_bibc = true_bibc_flat[:num_branches * num_non_slack].reshape(num_branches, num_non_slack)
    true_bcbv = true_bcbv_flat[:num_non_slack * num_branches].reshape(num_non_slack, num_branches)
    return true_bibc, true_bcbv, row['topology_id']

# -------------------------------------------------------------------
# 4. MAIN
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict BIBC and BCBV from V/I values.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample index from test data to display true vs predicted (0-based)")
    parser.add_argument("--vi", default="vi_test.csv", help="Test V/I CSV file (optional)")
    parser.add_argument("--target", default="bibc_bcbv_test.csv", help="Test target CSV file (optional)")
    parser.add_argument("--bcbv-method", default="median", choices=["median", "mean", "diagonal"],
                        help="How to reconcile a BCBV column's active entries into one consensus value")
    parser.add_argument("--match-tol", type=float, default=1e-4,
                        help="Max per-feature tolerance when matching pasted interactive values against test CSV")
    args = parser.parse_args()

    model, x_scaler, bcbv_scaler = load_model_and_scalers()

    # ---------- Sample mode ----------
    if args.sample is not None:
        try:
            vi_test = pd.read_csv(args.vi, dtype={'system_id': str})
            target_test = pd.read_csv(args.target, dtype={'system_id': str})
            df = vi_test.merge(target_test, on=['system_id', 'topology_id'])
            if len(df) == 0:
                raise FileNotFoundError
            if args.sample >= len(df):
                print(f"Sample index {args.sample} out of range (0..{len(df)-1})")
                sys.exit(1)
            row = df.iloc[args.sample]
            system_id = row['system_id']
            topo_id = row['topology_id']
            feature_cols = [c for c in vi_test.columns if c.startswith('V') or c.startswith('I')]
            target_cols = [c for c in target_test.columns if c.startswith('BIBC') or c.startswith('BCBV')]
            vals = row[feature_cols].values.astype(float).tolist()
            true_bibc_padded = row[target_cols[:289]].values.reshape(17, 17)
            true_bcbv_padded = row[target_cols[289:]].values.reshape(17, 17)
            num_buses, num_branches = get_dims(system_id)
            num_non_slack = num_buses - 1
            true_bibc = true_bibc_padded[:num_branches, :num_non_slack]
            true_bcbv = true_bcbv_padded[:num_non_slack, :num_branches]
            pred_bibc, pred_bcbv = predict_sample(model, x_scaler, bcbv_scaler, vals, system_id,
                                                   bcbv_method=args.bcbv_method)

            print(f"\n--- UNSEEN TEST DATA: SAMPLE {args.sample} ---")
            print(f"System: {system_id}-bus")
            print(f"Topology ID: {topo_id}")
            print(f"\n--- BIBC MATRIX ({num_branches}x{num_non_slack}) ---")
            print_comparison(true_bibc, pred_bibc, true_bcbv, pred_bcbv,
                             num_branches, num_non_slack, args.bcbv_method)
            sys.exit(0)
        except Exception as e:
            print(f"Error loading test data: {e}")
            print("Falling back to interactive mode.")

    # ---------- Interactive mode ----------
    print("\n" + "=" * 80)
    print("Interactive Prediction Mode")
    print("Paste numbers (V1..Vn, I1..Im) separated by spaces or commas.")
    print("The system is auto-detected from the number of non-zero voltages.")
    print("If the pasted values match a row in the test CSVs, the true BIBC/BCBV")
    print("and accuracy will be shown alongside the prediction.")
    print("Press Enter on an empty line to submit.")
    print("Type 'q' on a new line to quit.")
    print("=" * 80)

    df_info = load_test_dataframes(args.vi, args.target)

    while True:
        full_input = read_multi_line_input("\nPaste numbers (or 'q'):")
        if full_input is None:
            break
        if not full_input:
            continue
        parts = full_input.replace(',', ' ').split()
        try:
            vals = [float(x) for x in parts]
            if len(vals) == 0:
                continue
            system_id = detect_system_from_values(vals)
            pred_bibc, pred_bcbv = predict_sample(model, x_scaler, bcbv_scaler, vals, system_id,
                                                   bcbv_method=args.bcbv_method)
            print(f"\nSystem: {system_id}-bus")

            # Try to match ground truth
            match = find_ground_truth(vals, system_id, df_info, tol=args.match_tol)
            if match is not None:
                true_bibc, true_bcbv, topo_id = match
                num_buses, num_branches = get_dims(system_id)
                num_non_slack = num_buses - 1
                print(f"Matched test topology ID: {topo_id}")
                print_comparison(true_bibc, pred_bibc, true_bcbv, pred_bcbv,
                                 num_branches, num_non_slack, args.bcbv_method)
            else:
                print("(No matching row found in the test CSVs -- showing prediction only.)")
                print_predicted_only(pred_bibc, pred_bcbv)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()