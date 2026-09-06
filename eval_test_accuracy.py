"""
Quick bulk evaluation on the test set to get topology-match accuracy and BIBC element accuracy.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

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

def bibc_to_edges(bibc_mat):
    num_branches, num_non_slack = bibc_mat.shape
    edges = set()
    for i in range(num_branches):
        t = i + 2
        path_t = bibc_mat[:, i]
        if np.sum(path_t) == 1 and path_t[i] == 1:
            edges.add((1, t))
        elif path_t[i] == 1:
            target_path = path_t.copy(); target_path[i] = 0
            for col_f in range(num_non_slack):
                if np.array_equal(bibc_mat[:, col_f], target_path):
                    edges.add((col_f + 2, t)); break
    return edges

print("Loading model and scalers...")
model = Regressor()
model.load_state_dict(torch.load("unified_best.pth", map_location="cpu"))
model.eval()
x_scaler = joblib.load("x_scaler.pkl")
bcbv_scaler = joblib.load("bcbv_scaler.pkl")

print("Loading test CSVs...")
vi_test   = pd.read_csv("vi_test.csv", dtype={'system_id': str})
tgt_test  = pd.read_csv("bibc_bcbv_test.csv", dtype={'system_id': str})
df = vi_test.merge(tgt_test, on=['system_id','topology_id'])
df['system_id'] = df['system_id'].str.replace('.0','',regex=False)

vi_cols  = [c for c in vi_test.columns  if c.startswith('V') or c.startswith('I')]
tgt_cols = [c for c in tgt_test.columns if c.startswith('BIBC') or c.startswith('BCBV')]

dims = {'5':(5,4), '15':(15,14), '18':(18,17)}

# --- Topology-level accuracy: per unique topology, pick 1 representative sample ---
# (one VI sample per topology is enough to predict BIBC)
topo_groups = df.groupby(['system_id','topology_id'])

total_topos = 0
correct_topos = 0
total_bibc_elem = 0
correct_bibc_elem = 0
total_bcbv_mae_list = []

print(f"Evaluating {len(topo_groups)} unique test topologies...")
for (sys_id, topo_id), grp in topo_groups:
    row = grp.iloc[0]
    num_buses, num_branches = dims[str(sys_id)]
    num_non_slack = num_buses - 1

    vals = row[vi_cols].values.astype(np.float32)
    X_s = x_scaler.transform(vals.reshape(1,-1))
    X_t = torch.tensor(X_s, dtype=torch.float32)

    with torch.no_grad():
        out = model(X_t)
        bibc_probs = torch.sigmoid(out[0,:289])
        bibc_pred  = (bibc_probs > 0.5).int().numpy()
        bcbv_pred  = bcbv_scaler.inverse_transform(out[0,289:].numpy().reshape(1,-1)).flatten()

    true_bibc_flat = row[tgt_cols[:289]].values.astype(float)
    true_bibc = true_bibc_flat[:num_branches*num_non_slack].reshape(num_branches, num_non_slack)
    pred_bibc = bibc_pred[:num_branches*num_non_slack].reshape(num_branches, num_non_slack)

    # Element-level BIBC accuracy
    total_bibc_elem  += num_branches * num_non_slack
    correct_bibc_elem += int(np.sum(true_bibc == pred_bibc))

    # Topology-match (edge-set equality)
    true_edges = bibc_to_edges(true_bibc)
    pred_edges = bibc_to_edges(pred_bibc)
    total_topos += 1
    if true_edges == pred_edges:
        correct_topos += 1

bibc_elem_acc  = correct_bibc_elem / total_bibc_elem * 100
topo_match_acc = correct_topos / total_topos * 100

print("\n" + "="*55)
print("TEST SET RESULTS (best model, disjoint topologies)")
print("="*55)
print(f"Total test topologies evaluated : {total_topos}")
print(f"BIBC element-level accuracy     : {bibc_elem_acc:.2f}%")
print(f"Topology-match accuracy         : {topo_match_acc:.2f}%  ({correct_topos}/{total_topos})")
print("="*55)
