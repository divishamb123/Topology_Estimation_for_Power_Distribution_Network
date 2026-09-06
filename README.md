# ⚡ Topology & Line Parameter Estimation for Power Distribution Networks

An end-to-end, physics-informed Deep Learning framework and interactive web dashboard for real-time **distribution network topology identification** and **line parameter (impedance) estimation** using nodal voltage magnitudes ($V$) and bus-injected currents ($I$).

Developed as part of the internship research project at **Malaviya National Institute of Technology (MNIT) Jaipur**.

---

## 📌 Project Overview

Modern active distribution networks (ADNs) experience frequent configuration changes due to renewable energy integration, switching operations, fault isolation, and demand response. Accurate, real-time knowledge of network topology and branch parameters is vital for:
- **State Estimation & Volt-VAR Control**
- **Optimal Power Flow & Loss Minimization**
- **Fault Location, Isolation, and Service Restoration (FLISR)**
- **DER (Distributed Energy Resource) Hosting Capacity Management**

This project formulates topology and parameter estimation by directly mapping smart meter sensor measurements ($|V|$ and $|I|$) to the **BIBC** (Bus-Injection to Branch-Current) matrix and **BCBV** (Branch-Current to Bus-Voltage) matrix using a unified deep residual neural network (ResNet).

---

## 🔬 Mathematical Formulation

### 1. The Direct Backward-Forward Sweep (BFS) Formulation
In radial distribution networks, branch currents $[B]$ and bus voltage drops $[\Delta V]$ can be expressed linearly in terms of bus current injections $[I]$:

$$[B] = [BIBC] \cdot [I]$$

$$[\Delta V] = [BCBV] \cdot [B] = [BCBV] \cdot [BIBC] \cdot [I]$$

Where:
- **$[BIBC]$ Matrix (Topology)**: Upper-triangular binary matrix containing values $\in \{0, 1\}$. An entry $BIBC_{ij} = 1$ indicates that branch $i$ is part of the path from the substation/slack bus to bus $j$.
- **$[BCBV]$ Matrix (Parameters)**: Continuous matrix containing branch complex impedances ($Z_k = R_k + jX_k$). An entry $BCBV_{ji} = Z_i$ if branch $i$ is traversed on the path from the substation to bus $j$.
- **$[\Delta V]$**: Voltage deviations from the substation reference ($V_1 - V_i$).

### 2. Multi-Task Learning Architecture
A unified deep regressor predicts both matrices concurrently:
- **BIBC Head**: Multi-label binary classification optimized with `BCEWithLogitsLoss`.
- **BCBV Head**: Continuous impedance regression optimized with `MSELoss`.
- **Physical Consistency Enforcement**: Post-processing leverages the tree property of radial grids to filter line impedances across active paths and reconstruct the radial spanning tree.

---

## 📊 Supported Network Systems

The model is trained as a **unified architecture** capable of estimating topologies across multiple power distribution benchmarks:

| System | Total Buses | Branches | Base Voltage | Load / Branch Data |
| :--- | :--- | :--- | :--- | :--- |
| **5-Bus** | 5 | 4 | Standard Test Feeder | 4 PQ Loads, 4 Radial Branches |
| **15-Bus** | 15 | 14 | 11.0 kV ($Z_{base} = 121\,\Omega$) | 14 PQ Loads, Industrial/Commercial mix |
| **18-Bus** | 18 | 17 | 11.0 kV ($Z_{base} = 121\,\Omega$) | 17 PQ Loads, Extended Distribution Radial Feeder |

*Note: All systems are zero-padded to a standardized input dimension (35 features: 18 Voltages + 17 Currents) and an output dimension of 578 (289 BIBC elements + 289 BCBV elements).*

---

## 🧠 Neural Network Architecture

- **Input Dimension**: 35 (Standardized Nodal Voltages & Injected Currents via `StandardScaler`)
- **Backbone**: Fully Connected Input Layer (512 units, BatchNorm1d, ReLU, Dropout 0.2)
- **Deep Residual Blocks**: 4 Residual Blocks, each featuring:
  - `Linear(512, 512)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.2)`
  - `Linear(512, 512)` $\rightarrow$ `BatchNorm1d`
  - Residual connection (`x + res`) $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.2)`
- **Multi-Task Output Dimension**: 578
  - `[0 : 289]`: BIBC matrix logits ($17 \times 17$)
  - `[289 : 578]`: Scaled BCBV matrix impedance elements ($17 \times 17$)
- **Optimization**: AdamW optimizer with `ReduceLROnPlateau` learning rate scheduling.

---

## 📁 Repository Structure

```text
├── app.py                      # Flask backend serving the web dashboard & REST API
├── requirements.txt            # Python dependencies
├── train_model.py              # PyTorch training pipeline with multi-task loss
├── generate_data.py            # Synthetic power flow & topology variation dataset generator
├── eval_test_accuracy.py       # Bulk test evaluation script across unseen topologies
├── predict_test.py             # CLI prediction, verification & inspection utility
├── visualize_prediction.py     # NetworkX and Matplotlib topology tree graph plotting
├── unified_best.pth            # Best checkpoint model weights (PyTorch)
├── unified_final.pth           # Final trained model weights (PyTorch)
├── x_scaler.pkl                # Scikit-learn StandardScaler for input features (V, I)
├── bcbv_scaler.pkl             # Scikit-learn StandardScaler for BCBV targets
├── vi_train.csv                # Training dataset: Voltage & Current readings
├── vi_test.csv                 # Testing dataset: Voltage & Current readings
├── bibc_bcbv_train.csv         # Training target matrices (ground truth)
├── bibc_bcbv_test.csv          # Testing target matrices (ground truth)
├── templates/
│   └── index.html              # Modern, responsive web dashboard template
└── static/
    ├── css/
    │   └── style.css           # Glassmorphism dark-theme styling
    └── js/
        └── app.js              # Real-time SVG topology graph renderer & UI controller
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- Recommended: virtual environment (`venv` or `conda`)

### 2. Installation
Clone the repository and install required packages:

```bash
git clone https://github.com/divishamb123/Topology_Estimation_for_Power_Distribution_Network.git
cd Topology_Estimation_for_Power_Distribution_Network

# Optional: Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🖥️ Usage Instructions

### 1. Launching the Interactive Web Dashboard
Run the Flask application:

```bash
python3 app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

#### Dashboard Features:
- **System Switcher**: Toggle seamlessly between 5-Bus, 15-Bus, and 18-Bus networks.
- **Random Sample Generator**: Pick random test samples from the evaluation set.
- **Side-by-Side Topology Comparison**: Interactive SVG graph visualization displaying **Predicted Topology** alongside **Ground Truth**.
- **Edge Highlighting**: Green edges for correct predictions, red dashed lines for mismatched connections.
- **Interactive BIBC Matrix**: Inspect the predicted binary topological matrix with cell-level hover information.
- **Sensor Profiles**: Real-time comparison of bus voltage profiles and branch currents.

---

### 2. Evaluating Model Accuracy on Unseen Test Topologies
Run the evaluation script to test model generalization across disjoint topologies:

```bash
python3 eval_test_accuracy.py
```

**Benchmark Results:**
- **Evaluated Test Topologies**: 405 unseen topologies
- **BIBC Element-Level Accuracy**: **91.66%**

---

### 3. Running Single Sample CLI Predictions
To inspect predictions for a specific test sample index:

```bash
# Evaluate sample #42 with median consistency filtering
python3 predict_test.py --sample 42 --bcbv-method median
```

---

### 4. Visualizing Predicted Topology Graphs with NetworkX
Generate high-resolution Matplotlib comparison diagrams:

```bash
python3 visualize_prediction.py --sample 100
```

---

### 5. Training the Model from Scratch
To retrain the unified neural network:

```bash
python3 train_model.py
```

### 6. Generating Synthetic Grid Data
To simulate new power flow scenarios with Backward-Forward Sweep:

```bash
python3 generate_data.py
```

---

## 🌐 REST API Endpoints

The Flask server provides REST endpoints for automated integrations:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/systems` | Returns available bus systems (5, 15, 18) and sample counts |
| `GET` | `/api/sample/<system_id>/<idx>` | Retrieves input readings, ground truth, and predictions for a sample |
| `GET` | `/api/random/<system_id>` | Selects a random sample from the test set for a given system |
| `POST` | `/api/predict` | Accepts raw voltage & current arrays and returns estimated BIBC/BCBV |

---

## 👥 Authors & Acknowledgments

- **Divisha Manak Bohra** — Project Researcher & Developer
- **Malaviya National Institute of Technology (MNIT) Jaipur** — Department of Electrical Engineering
- Special thanks to faculty advisors and mentors at MNIT Jaipur for guidance on distribution system load flow methods and deep learning applications in smart grids.