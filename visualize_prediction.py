
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import argparse
import sys
import matplotlib.pyplot as plt
import networkx as nx

np.set_printoptions(threshold=np.inf, suppress=True, precision=4, linewidth=200)

# ==========================================
# 1. PYTORCH MODEL DEFINITION
# ==========================================
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

# ==========================================
# 2. VISUALIZATION & GRAPH FUNCTIONS
# ==========================================
def plot_vi_profiles(vals, system_id):
    """Plots the input Voltage and Current sensor readings."""
    num_buses, num_branches = get_dims(system_id)
    V_mag = vals[:num_buses]
    I_mag = vals[18:18+num_branches]
    
    buses = range(1, num_buses + 1)
    branches = range(1, num_branches + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Input Sensor Data ({system_id}-Bus System)', fontsize=14, fontweight='bold')

    axes[0].plot(buses, V_mag, marker='o', linewidth=2, color='blue')
    axes[0].set_title("Bus Voltage Profile")
    axes[0].set_xlabel("Bus Number")
    axes[0].set_ylabel("Voltage Magnitude (p.u.)")
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].set_xticks(buses)

    axes[1].bar(branches, I_mag, color='orange', edgecolor='black')
    axes[1].set_title("Branch Current Magnitude")
    axes[1].set_xlabel("Branch Number")
    axes[1].set_ylabel("Current Magnitude")
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.7)
    axes[1].set_xticks(branches)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show(block=False) 

def plot_matrix_comparison(true_bibc, pred_bibc, true_bcbv, pred_bcbv, system_id):
    """Plots side-by-side heatmaps of the True vs Predicted matrices."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'AI Matrix Prediction Results ({system_id}-Bus System)', fontsize=16, fontweight='bold')

    im1 = axes[0, 0].imshow(true_bibc, cmap='viridis', aspect='auto')
    axes[0, 0].set_title("True BIBC Matrix", fontsize=12)
    fig.colorbar(im1, ax=axes[0, 0], label="Connection (0 or 1)")

    im2 = axes[0, 1].imshow(pred_bibc, cmap='viridis', aspect='auto')
    axes[0, 1].set_title("Predicted BIBC Matrix", fontsize=12)
    fig.colorbar(im2, ax=axes[0, 1], label="Connection (0 or 1)")

    im3 = axes[1, 0].imshow(np.abs(true_bcbv), cmap='plasma', aspect='auto')
    axes[1, 0].set_title("True BCBV Matrix (Ohms)", fontsize=12)
    fig.colorbar(im3, ax=axes[1, 0], label="Impedance Magnitude")

    im4 = axes[1, 1].imshow(np.abs(pred_bcbv), cmap='plasma', aspect='auto')
    axes[1, 1].set_title("Predicted BCBV Matrix (Ohms)", fontsize=12)
    fig.colorbar(im4, ax=axes[1, 1], label="Impedance Magnitude")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92) 
    plt.show(block=False)

def bibc_to_edges(bibc_mat):
    """Reverse-engineers a BIBC matrix back into physical graph edges."""
    num_branches, num_non_slack = bibc_mat.shape
    edges = []
    
    for i in range(num_branches):
        t = i + 2 
        path_t = bibc_mat[:, i] 
        
        if np.sum(path_t) == 1 and path_t[i] == 1:
            edges.append((1, t))
        elif path_t[i] == 1:
            target_path = path_t.copy()
            target_path[i] = 0
            for col_f in range(num_non_slack):
                if np.array_equal(bibc_mat[:, col_f], target_path):
                    f = col_f + 2 
                    edges.append((f, t))
                    break
    return edges

def plot_topology_graph(true_bibc, pred_bibc, system_id):
    """Draws the physical node-and-branch layout of the power grid without overlapping nodes."""
    true_edges = bibc_to_edges(true_bibc)
    pred_edges = bibc_to_edges(pred_bibc)

    # 1. Increased figsize for a wider canvas
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Physical Grid Topology ({system_id}-Bus System)', fontsize=16, fontweight='bold')

    # --- True Topology Graph ---
    G_true = nx.DiGraph()
    G_true.add_edges_from(true_edges)
    
    # 2. Changed to spring_layout. 
    # 'k' dictates the distance between nodes (higher = further apart).
    # 'seed' ensures the graph looks the exact same way every time you run it.
    pos_true = nx.spring_layout(G_true, k=0.9, iterations=100, seed=42) 
    
    # 3. Slightly reduced node_size (600) and font_size (10) to reduce clutter
    nx.draw(G_true, pos_true, ax=axes[0], with_labels=True, node_color='lightblue', 
            node_size=600, font_size=10, font_weight='bold', edge_color='gray', arrows=True)
    axes[0].set_title("True Topology Layout", fontsize=14)

    # --- Predicted Topology Graph ---
    G_pred = nx.DiGraph()
    G_pred.add_edges_from(pred_edges)
    
    if set(G_true.nodes()) == set(G_pred.nodes()):
        pos_pred = pos_true
    else:
        pos_pred = nx.spring_layout(G_pred, k=0.9, iterations=100, seed=42)
        
    nx.draw(G_pred, pos_pred, ax=axes[1], with_labels=True, node_color='#ffcccb', 
            node_size=600, font_size=10, font_weight='bold', edge_color='red', arrows=True)
    axes[1].set_title("AI Predicted Topology Layout", fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show() # This blocks and keeps all windows open


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
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
        consensus = np.median(active_vals) if method == 'median' else np.mean(active_vals)
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

# ==========================================
# 4. EVALUATION & MAIN LOGIC
# ==========================================
def print_comparison(true_bibc, pred_bibc, true_bcbv_raw, pred_bcbv, num_branches, num_non_slack, bcbv_method='median'):
    true_bcbv = prepare_true_bcbv(true_bibc, true_bcbv_raw, bcbv_method)
    acc = np.sum(true_bibc == pred_bibc)
    total = num_branches * num_non_slack
    print(f"Accuracy: {acc}/{total} correct ({acc/total*100:.1f}%)")
    
    true_mask = true_bibc.T
    active = true_mask > 0.5
    if np.any(active):
        mae = np.mean(np.abs(pred_bcbv[active] - true_bcbv[active]))
        print(f"\nMean Absolute Error (on active branches): {mae:.6f} ohms")
    return true_bcbv 

def main():
    parser = argparse.ArgumentParser(description="Evaluate BIBC and BCBV with Visualizations.")
    parser.add_argument("--sample", type=int, default=None, help="Sample index to test (e.g., 42).")
    parser.add_argument("--vi-test", default="vi_test.csv", help="Testing V/I CSV file")
    parser.add_argument("--target-test", default="bibc_bcbv_test.csv", help="Testing target CSV file")
    parser.add_argument("--bcbv-method", default="median", choices=["median", "mean", "diagonal"])
    args = parser.parse_args()

    model, x_scaler, bcbv_scaler = load_model_and_scalers()

    try:
        vi_test = pd.read_csv(args.vi_test, dtype={'system_id': str})
        target_test = pd.read_csv(args.target_test, dtype={'system_id': str})
        df = vi_test.merge(target_test, on=['system_id', 'topology_id'])
        
        if len(df) == 0:
            print("Error: The merged dataframe is empty.")
            sys.exit(1)
            
        vi_cols = [c for c in vi_test.columns if c.startswith('V') or c.startswith('I')]
        target_cols = [c for c in target_test.columns if c.startswith('BIBC') or c.startswith('BCBV')]

        if args.sample is not None:
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

            print(f"\n--- TESTING DATA: SAMPLE {args.sample} ---")
            print(f"System: {system_id}-bus")
            print(f"Topology ID: {topo_id}")
            
            true_bcbv_prepared = print_comparison(true_bibc, pred_bibc, true_bcbv_raw, pred_bcbv, num_branches, num_non_slack, args.bcbv_method)
            
            # Trigger Visualizations
            print("\nGenerating visual plots... (Close the plot windows to exit the script)")
            plot_vi_profiles(vals, system_id)
            plot_matrix_comparison(true_bibc, pred_bibc, true_bcbv_prepared, pred_bcbv, system_id)
            plot_topology_graph(true_bibc, pred_bibc, system_id)
            
        else:
            print("Please specify a sample to visualize by running: python final_prediction.py --sample <number>")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()