"""
Power Grid Topology Identifier — Web Dashboard Backend
Flask server wrapping the trained PyTorch model and serving predictions via REST API.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from flask import Flask, render_template, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))

# ── Model Definition (must match training) ─────────────────────
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
    def __init__(self, input_dim=35, hidden=512, output_dim=578,
                 num_blocks=4, dropout=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden, dropout) for _ in range(num_blocks)]
        )
        self.output_layer = nn.Linear(hidden, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.output_layer(x)


# ── Load Model & Data ─────────────────────────────────────────
print("Loading model and scalers...")
model = Regressor()
model.load_state_dict(
    torch.load(os.path.join(BASE_DIR, "unified_final.pth"), map_location="cpu")
)
model.eval()
x_scaler = joblib.load(os.path.join(BASE_DIR, "x_scaler.pkl"))
bcbv_scaler = joblib.load(os.path.join(BASE_DIR, "bcbv_scaler.pkl"))

print("Loading test data...")
vi_test = pd.read_csv(os.path.join(BASE_DIR, "vi_test.csv"),
                       dtype={'system_id': str})
target_test = pd.read_csv(os.path.join(BASE_DIR, "bibc_bcbv_test.csv"),
                           dtype={'system_id': str})
df_test = vi_test.merge(target_test, on=['system_id', 'topology_id'])
df_test['system_id'] = (df_test['system_id'].astype(str)
                        .str.replace('.0', '', regex=False))

vi_cols = [c for c in vi_test.columns if c.startswith('V') or c.startswith('I')]
target_cols = [c for c in target_test.columns
               if c.startswith('BIBC') or c.startswith('BCBV')]
print(f"Ready — {len(df_test)} test samples loaded.")


# ── Helper Functions ──────────────────────────────────────────
def get_dims(system_id):
    dims = {'5': (5, 4), '15': (15, 14), '18': (18, 17)}
    return dims[str(system_id)]


def enforce_bcbv_consistency(bcbv_mat, active_mask):
    bcbv_mat = bcbv_mat.copy()
    _, num_branches = bcbv_mat.shape
    for j in range(num_branches):
        col_active = active_mask[:, j] > 0
        if not np.any(col_active):
            continue
        bcbv_mat[col_active, j] = np.median(bcbv_mat[col_active, j])
    return bcbv_mat


def predict(vals, system_id):
    """Run AI prediction → returns (bibc_mat, bcbv_mat)."""
    num_buses, num_branches = get_dims(system_id)
    num_non_slack = num_buses - 1
    vals = list(vals)
    if len(vals) < 35:
        vals += [0.0] * (35 - len(vals))

    X = np.array(vals[:35], dtype=np.float32).reshape(1, -1)
    X_scaled = x_scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        out = model(X_t)
        bibc_probs = torch.sigmoid(out[0, :289])
        bibc_pred = (bibc_probs > 0.5).int().numpy()
        bcbv_pred = bcbv_scaler.inverse_transform(
            out[0, 289:].numpy().reshape(1, -1)
        ).flatten()

    bibc_mat = bibc_pred[:num_branches * num_non_slack].reshape(
        num_branches, num_non_slack)
    bcbv_mat = bcbv_pred[:num_non_slack * num_branches].reshape(
        num_non_slack, num_branches)

    active_mask = bibc_mat.T
    bcbv_mat = bcbv_mat * active_mask
    bcbv_mat = np.where(np.abs(bcbv_mat) < 1e-10, 0.0, bcbv_mat)
    bcbv_mat = enforce_bcbv_consistency(bcbv_mat, active_mask)
    return bibc_mat, bcbv_mat


def prepare_true_bcbv(true_bibc, true_bcbv_raw):
    true_mask = true_bibc.T
    bcbv = true_bcbv_raw * true_mask
    bcbv = np.where(np.abs(bcbv) < 1e-10, 0.0, bcbv)
    return enforce_bcbv_consistency(bcbv, true_mask)


def bibc_to_edges(bibc_mat):
    """Reverse-engineer BIBC matrix → list of [from, to] edges."""
    num_branches, num_non_slack = bibc_mat.shape
    edges = []
    for i in range(num_branches):
        t = i + 2
        path_t = bibc_mat[:, i]
        if np.sum(path_t) == 1 and path_t[i] == 1:
            edges.append([1, t])
        elif path_t[i] == 1:
            target_path = path_t.copy()
            target_path[i] = 0
            for col_f in range(num_non_slack):
                if np.array_equal(bibc_mat[:, col_f], target_path):
                    edges.append([col_f + 2, t])
                    break
    return edges


# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/systems')
def get_systems():
    systems = {}
    for sys_id in ['5', '15', '18']:
        sub = df_test[df_test['system_id'] == sys_id]
        nb, nbr = get_dims(sys_id)
        systems[sys_id] = {
            'num_samples': len(sub),
            'num_buses': nb,
            'num_branches': nbr
        }
    return jsonify(systems)


@app.route('/api/sample/<system_id>/<int:idx>')
def get_sample(system_id, idx):
    sub = df_test[df_test['system_id'] == system_id].reset_index(drop=True)
    if len(sub) == 0:
        return jsonify({'error': f'No data for system {system_id}'}), 404
    if idx < 0 or idx >= len(sub):
        return jsonify({'error': f'Index {idx} out of range (0..{len(sub)-1})'}), 400

    row = sub.iloc[idx]
    vals = row[vi_cols].values.astype(float)
    true_bibc_flat = row[target_cols[:289]].values.astype(float)
    true_bcbv_flat = row[target_cols[289:]].values.astype(float)

    num_buses, num_branches = get_dims(system_id)
    num_non_slack = num_buses - 1

    true_bibc = true_bibc_flat[:num_branches * num_non_slack].reshape(
        num_branches, num_non_slack)
    true_bcbv_raw = true_bcbv_flat[:num_non_slack * num_branches].reshape(
        num_non_slack, num_branches)
    true_bcbv = prepare_true_bcbv(true_bibc, true_bcbv_raw)

    pred_bibc, pred_bcbv = predict(vals, system_id)

    # Metrics
    bibc_correct = int(np.sum(true_bibc == pred_bibc))
    bibc_total = num_branches * num_non_slack
    bibc_acc = round(bibc_correct / bibc_total * 100, 2)

    true_mask = true_bibc.T
    active = true_mask > 0.5
    bcbv_mae = (round(float(np.mean(np.abs(
        pred_bcbv[active] - true_bcbv[active]))), 6)
        if np.any(active) else 0.0)

    true_edges = bibc_to_edges(true_bibc)
    pred_edges = bibc_to_edges(pred_bibc)
    topo_match = set(map(tuple, true_edges)) == set(map(tuple, pred_edges))

    return jsonify({
        'system_id':      system_id,
        'sample_index':   idx,
        'topology_id':    int(row['topology_id']),
        'num_buses':      num_buses,
        'num_branches':   num_branches,
        'num_samples':    len(sub),
        'voltages':       vals[:num_buses].tolist(),
        'currents':       vals[18:18 + num_branches].tolist(),
        'true_bibc':      true_bibc.astype(int).tolist(),
        'pred_bibc':      pred_bibc.astype(int).tolist(),
        'true_bcbv':      np.round(true_bcbv, 4).tolist(),
        'pred_bcbv':      np.round(pred_bcbv, 4).tolist(),
        'bibc_accuracy':  bibc_acc,
        'bibc_correct':   bibc_correct,
        'bibc_total':     bibc_total,
        'bcbv_mae':       bcbv_mae,
        'true_edges':     true_edges,
        'pred_edges':     pred_edges,
        'topology_match': topo_match,
    })


@app.route('/api/random/<system_id>')
def get_random(system_id):
    sub = df_test[df_test['system_id'] == system_id]
    if len(sub) == 0:
        return jsonify({'error': f'No data for system {system_id}'}), 404
    idx = int(np.random.randint(0, len(sub)))
    return get_sample(system_id, idx)


def auto_detect_system(vals, fallback='5'):
    vals = list(vals)
    n = len(vals)

    if n == 9:
        # 5 voltages + 4 currents
        formatted = vals[:5] + [0.0] * 13 + vals[5:9] + [0.0] * 13
        return '5', formatted
    elif n == 29:
        # 15 voltages + 14 currents
        formatted = vals[:15] + [0.0] * 3 + vals[15:29] + [0.0] * 3
        return '15', formatted

    if n >= 18:
        voltages = vals[:18]
        nz_v = sum(1 for v in voltages if abs(v) > 1e-5)
        if nz_v <= 5:
            return '5', vals
        elif nz_v <= 15:
            return '15', vals
        else:
            return '18', vals

    return fallback, vals


def find_ground_truth_for_vals(vals, system_id, tol=1e-3):
    sub = df_test[df_test['system_id'] == str(system_id)].reset_index(drop=True)
    if len(sub) == 0:
        return None
    feat_matrix = sub[vi_cols].values.astype(float)
    n_feat = len(vi_cols)
    padded = list(vals) + [0.0] * max(0, n_feat - len(vals))
    padded = np.array(padded[:n_feat], dtype=float)
    diffs = np.max(np.abs(feat_matrix - padded), axis=1)
    best_idx = int(np.argmin(diffs))
    if diffs[best_idx] <= tol:
        return sub.iloc[best_idx]
    return None


@app.route('/api/predict', methods=['POST'])
def predict_custom():
    data = request.json
    vals = data.get('values', [])
    requested_sys = str(data.get('system_id', '5'))

    if not vals:
        return jsonify({'error': 'Missing values'}), 400

    system_id, vals = auto_detect_system(vals, fallback=requested_sys)

    try:
        num_buses, num_branches = get_dims(system_id)
        num_non_slack = num_buses - 1
    except Exception:
        return jsonify({'error': f'Invalid system_id: {system_id}'}), 400

    pred_bibc, pred_bcbv = predict(vals, system_id)
    pred_edges = bibc_to_edges(pred_bibc)

    V = (vals[:num_buses] if len(vals) >= num_buses
         else vals + [0.0] * (num_buses - len(vals)))
    I_start = 18 if len(vals) >= 35 else num_buses
    I = (vals[I_start:I_start + num_branches]
         if len(vals) > I_start else [0.0] * num_branches)

    res = {
        'system_id':    system_id,
        'num_buses':    num_buses,
        'num_branches': num_branches,
        'voltages':     V,
        'currents':     I,
        'pred_bibc':    pred_bibc.astype(int).tolist(),
        'pred_bcbv':    np.round(pred_bcbv, 4).tolist(),
        'pred_edges':   pred_edges,
        'custom_input': True,
    }

    # Try matching ground truth in test dataset
    matched_row = find_ground_truth_for_vals(vals, system_id)
    if matched_row is not None:
        true_bibc_flat = matched_row[target_cols[:289]].values.astype(float)
        true_bcbv_flat = matched_row[target_cols[289:]].values.astype(float)

        true_bibc = true_bibc_flat[:num_branches * num_non_slack].reshape(
            num_branches, num_non_slack)
        true_bcbv_raw = true_bcbv_flat[:num_non_slack * num_branches].reshape(
            num_non_slack, num_branches)
        true_bcbv = prepare_true_bcbv(true_bibc, true_bcbv_raw)

        bibc_correct = int(np.sum(true_bibc == pred_bibc))
        bibc_total = num_branches * num_non_slack
        bibc_acc = round(bibc_correct / bibc_total * 100, 2)

        true_mask = true_bibc.T
        active = true_mask > 0.5
        bcbv_mae = (round(float(np.mean(np.abs(
            pred_bcbv[active] - true_bcbv[active]))), 6)
            if np.any(active) else 0.0)

        true_edges = bibc_to_edges(true_bibc)
        topo_match = set(map(tuple, true_edges)) == set(map(tuple, pred_edges))

        res.update({
            'matched_ground_truth': True,
            'topology_id':    int(matched_row['topology_id']),
            'true_bibc':      true_bibc.astype(int).tolist(),
            'true_bcbv':      np.round(true_bcbv, 4).tolist(),
            'bibc_accuracy':  bibc_acc,
            'bibc_correct':   bibc_correct,
            'bibc_total':     bibc_total,
            'bcbv_mae':       bcbv_mae,
            'true_edges':     true_edges,
            'topology_match': topo_match,
        })

    return jsonify(res)


# ── Entry Point ───────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  ⚡ Power Grid Topology Identifier — Dashboard")
    print("  Open  http://localhost:5050  in your browser")
    print("=" * 60 + "\n")
    app.run(debug=False, port=5050, host='0.0.0.0')
