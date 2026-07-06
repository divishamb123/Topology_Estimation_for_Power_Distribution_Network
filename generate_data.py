import numpy as np
import csv
import random
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

MAX_BUSES = 18
MAX_BRANCHES = MAX_BUSES - 1
MAX_NON_SLACK = MAX_BUSES - 1

# ---------- 5-bus data ----------
rx_5 = [[0.0073,0.0025],[0.0009,0.0007],[0.0011,0.0008],[0.0009,0.0007]]
loads_5 = [(2,141,80),(3,89,50),(4,111,63),(5,140,80)]

# ---------- 15-bus data ----------
V_base_15 = 11.0
Z_base_15 = (V_base_15**2)/1.0  # 121 ohm
branches_15_raw = [
    (1,2,0.7766,0.7596), (2,3,0.6716,0.6569), (3,4,0.4827,0.4722),
    (4,5,0.8744,0.5898), (2,9,1.1554,0.7793), (9,10,0.9680,0.6529),
    (2,6,1.4677,0.9900), (6,7,0.6245,0.4213), (6,8,0.7182,0.4844),
    (3,11,1.0305,0.6951), (11,12,1.4052,0.9478), (12,13,1.1554,0.7793),
    (4,14,1.2803,0.8636), (4,15,0.6870,0.4634)
]
rx_15 = [[R/Z_base_15, X/Z_base_15] for (_,_,R,X) in branches_15_raw]
loads_15 = [
    (2,44.1,45),(3,70,71.4),(4,140,142.8),(5,44.1,45),
    (6,140,142.8),(7,140,142.8),(8,70,71.4),(9,70,71.4),
    (10,44.1,45),(11,140,142.8),(12,70,71.4),(13,44.1,45),
    (14,70,71.4),(15,140,142.8)
]

# ---------- 18-bus data ----------
V_base_18 = 11.0
Z_base_18 = (V_base_18**2)/1.0  # 121 ohm
branches_18_raw = [
    (1,2,0.7766,0.7596), (2,3,0.6716,0.6569), (3,4,0.4827,0.4722),
    (4,5,0.8744,0.5898), (2,9,1.1554,0.7793), (9,10,0.9680,0.6529),
    (2,6,1.4677,0.9900), (6,7,0.6245,0.4213), (6,8,0.7182,0.4844),
    (3,11,1.0305,0.6951), (11,12,1.4052,0.9478), (12,13,1.1554,0.7793),
    (4,14,1.2803,0.8636), (4,15,0.6870,0.4634), (5,16,0.6245,0.4213),
    (16,17,0.7182,0.4844), (17,18,0.7182,0.4844)
]
rx_18 = [[R/Z_base_18, X/Z_base_18] for (_,_,R,X) in branches_18_raw]
loads_18 = [
    (2,44.1,45),(3,70,71.4),(4,140,142.8),(5,44.1,45),
    (6,140,142.8),(7,140,142.8),(8,70,71.4),(9,70,71.4),
    (10,44.1,45),(11,140,142.8),(12,70,71.4),(13,44.1,45),
    (14,70,71.4),(15,140,142.8),(16,70,71.4),(17,70,71.4),(18,44.1,45)
]

SYSTEMS = {
    '5': {
        'num_buses': 5,
        'branch_rx': rx_5,
        'loads': loads_5,
        'base_kva': 1000.0,
        'n_train': 19,
        'n_test': 5,
        'generate_all': True,
    },
    '15': {
        'num_buses': 15,
        'branch_rx': rx_15,
        'loads': loads_15,
        'base_kva': 1000.0,
        'n_train': 1000,
        'n_test': 200,
        'generate_all': False,
    },
    '18': {
        'num_buses': 18,
        'branch_rx': rx_18,
        'loads': loads_18,
        'base_kva': 1000.0,
        'n_train': 1000,
        'n_test': 200,
        'generate_all': False,
    }
}

# Load flow engine
def build_BIBC(branches, num_buses):
    SLACK = 1
    num_non_slack = num_buses - 1
    num_br = len(branches)
    BIBC = [[0]*num_non_slack for _ in range(num_br)]
    for idx, (f,t,R,X) in enumerate(branches):
        col_to = t-2
        if f != SLACK:
            col_from = f-2
            for row in range(num_br):
                BIBC[row][col_to] = BIBC[row][col_from]
        BIBC[idx][col_to] = 1
    return BIBC

def build_BCBV(branches, num_buses):
    SLACK = 1
    num_non_slack = num_buses - 1
    num_br = len(branches)
    BCBV = [[0j]*num_br for _ in range(num_non_slack)]
    for idx, (f,t,R,X) in enumerate(branches):
        row_to = t-2
        if f != SLACK:
            row_from = f-2
            for col in range(num_br):
                BCBV[row_to][col] = BCBV[row_from][col]
        BCBV[row_to][idx] = complex(R,X)
    return BCBV

def run_loadflow(branches, loads_pu, num_buses, base_kva):
    try:
        num_br = len(branches)
        BIBC = build_BIBC(branches, num_buses)
        BCBV = build_BCBV(branches, num_buses)
        BIBC_np = np.array(BIBC, dtype=float)
        BCBV_np = np.array(BCBV, dtype=complex)
        DLF = BCBV_np @ BIBC_np

        S = np.zeros(num_buses+1, dtype=complex)
        for bus, P_pu, Q_pu in loads_pu:
            S[bus] += complex(P_pu, Q_pu)
        V = np.ones(num_buses+1, dtype=complex)
        V[1] = 1.0
        for _ in range(100):
            I_inj = np.conj(S[2:] / V[2:])
            deltaV = DLF @ I_inj
            V[2:] = 1.0 - deltaV
            if np.max(np.abs(deltaV)) < 1e-8:
                break
        V_mag = np.abs(V[1:])
        I_br = np.zeros(num_br)
        for i, (f,t,R,X) in enumerate(branches):
            Z = complex(R,X)
            I_br[i] = np.abs((V[f] - V[t]) / Z)
        if np.any(np.isnan(V_mag)) or np.any(np.isinf(V_mag)):
            return None, None
        return V_mag, I_br
    except:
        return None, None

def random_load_profile(loads, base_kva):
    loads_pu = []
    for bus, P_kw, Q_kvar in loads:
        factor = random.uniform(0.9, 1.1)
        loads_pu.append((bus, (P_kw*factor)/base_kva, (Q_kvar*factor)/base_kva))
    return loads_pu

def add_noise(vals, percent=1.0):
    return [v * (1 + np.random.normal(0, percent/100)) for v in vals]

def pad_flat(flat_list, target_size):
    return list(flat_list) + [0.0] * (target_size - len(flat_list))

def random_topology(rx_list):
    branches = []
    for j, (R,X) in enumerate(rx_list):
        from_bus = 1 if j==0 else random.randint(2, j+1)
        to_bus = j+2
        branches.append((from_bus, to_bus, R, X))
    return branches

def generate_all_topologies(rx_list, num_buses):
    def rec(current, next_bus):
        if next_bus > num_buses:
            return [current]
        res = []
        for parent in range(1, next_bus):
            R,X = rx_list[next_bus-2]
            new = current + [(parent, next_bus, R, X)]
            res.extend(rec(new, next_bus+1))
        return res
    return rec([], 2)

def generate_topologies(system_id, n_train, n_test):
    rx = SYSTEMS[system_id]['branch_rx']
    num_buses = SYSTEMS[system_id]['num_buses']
    config = SYSTEMS[system_id]

    if config.get('generate_all', False):
        tops = generate_all_topologies(rx, num_buses)
        random.shuffle(tops)
        return tops[:n_train], tops[n_train:n_train+n_test]
    else:
        seen = set()
        train_tops = []
        test_tops = []
        base_loads = config['loads']
        max_attempts = 200000

        # Train
        attempts = 0
        while len(train_tops) < n_train and attempts < max_attempts:
            br = random_topology(rx)
            key = tuple((f,t) for f,t,_,_ in br)
            if key in seen:
                attempts += 1
                continue
            seen.add(key)
            loads_pu = [(bus, P_kw/config['base_kva'], Q_kvar/config['base_kva']) for bus,P_kw,Q_kvar in base_loads]
            V, I = run_loadflow(br, loads_pu, num_buses, config['base_kva'])
            if V is not None:
                train_tops.append(br)
            attempts += 1
            if len(train_tops) % 200 == 0 and len(train_tops)>0:
                print(f"  Train: {len(train_tops)}/{n_train}")
        print(f"  Train valid: {len(train_tops)}")

        # Test
        attempts = 0
        while len(test_tops) < n_test and attempts < max_attempts:
            br = random_topology(rx)
            key = tuple((f,t) for f,t,_,_ in br)
            if key in seen:
                attempts += 1
                continue
            seen.add(key)
            loads_pu = [(bus, P_kw/config['base_kva'], Q_kvar/config['base_kva']) for bus,P_kw,Q_kvar in base_loads]
            V, I = run_loadflow(br, loads_pu, num_buses, config['base_kva'])
            if V is not None:
                test_tops.append(br)
            attempts += 1
            if len(test_tops) % 100 == 0 and len(test_tops)>0:
                print(f"  Test: {len(test_tops)}/{n_test}")
        print(f"  Test valid: {len(test_tops)}")
        return train_tops, test_tops

def main():
    random.seed(42)
    np.random.seed(42)
    n_profiles = 50

    vi_header = ["system_id","topology_id","profile_id"] + [f"V{b}" for b in range(1, MAX_BUSES+1)] + [f"I{br}" for br in range(1, MAX_BRANCHES+1)]
    bibc_cols = [f"BIBC_{r}_{c}" for r in range(1, MAX_BRANCHES+1) for c in range(1, MAX_NON_SLACK+1)]
    bcbv_cols = [f"BCBV_{r}_{c}" for r in range(1, MAX_NON_SLACK+1) for c in range(1, MAX_BRANCHES+1)]
    target_header = ["system_id","topology_id"] + bibc_cols + bcbv_cols

    vi_train = open("vi_train.csv", "w", newline="")
    vi_test = open("vi_test.csv", "w", newline="")
    target_train = open("bibc_bcbv_train.csv", "w", newline="")
    target_test = open("bibc_bcbv_test.csv", "w", newline="")

    csv.writer(vi_train).writerow(vi_header)
    csv.writer(vi_test).writerow(vi_header)
    csv.writer(target_train).writerow(target_header)
    csv.writer(target_test).writerow(target_header)

    for system_id in SYSTEMS:
        config = SYSTEMS[system_id]
        print(f"Generating for system {system_id}...")
        train_tops, test_tops = generate_topologies(
            system_id, config['n_train'], config['n_test']
        )
        print(f"  Train: {len(train_tops)}, Test: {len(test_tops)}")

        # Write train
        wrt_vi_train = csv.writer(vi_train)
        wrt_target_train = csv.writer(target_train)
        for topo_id, branches in enumerate(train_tops, start=1):
            BIBC = build_BIBC(branches, config['num_buses'])
            BCBV = build_BCBV(branches, config['num_buses'])
            flat_bibc = [int(v) for row in BIBC for v in row]
            flat_bcbv = [z.real for row in BCBV for z in row]
            bibc_pad = pad_flat(flat_bibc, MAX_BRANCHES * MAX_NON_SLACK)
            bcbv_pad = pad_flat(flat_bcbv, MAX_NON_SLACK * MAX_BRANCHES)
            wrt_target_train.writerow([system_id, topo_id] + bibc_pad + bcbv_pad)

            for prof_id in range(1, n_profiles+1):
                loads_pu = random_load_profile(config['loads'], config['base_kva'])
                V_true, I_true = run_loadflow(branches, loads_pu, config['num_buses'], config['base_kva'])
                if V_true is None:
                    continue
                V_noisy = add_noise(V_true, 1.0)
                I_noisy = add_noise(I_true, 1.0)
                V_pad = list(V_noisy) + [0.0] * (MAX_BUSES - config['num_buses'])
                I_pad = list(I_noisy) + [0.0] * (MAX_BRANCHES - len(branches))
                wrt_vi_train.writerow([system_id, topo_id, prof_id] + V_pad + I_pad)

        # Write test
        wrt_vi_test = csv.writer(vi_test)
        wrt_target_test = csv.writer(target_test)
        for topo_id, branches in enumerate(test_tops, start=len(train_tops)+1):
            BIBC = build_BIBC(branches, config['num_buses'])
            BCBV = build_BCBV(branches, config['num_buses'])
            flat_bibc = [int(v) for row in BIBC for v in row]
            flat_bcbv = [z.real for row in BCBV for z in row]
            bibc_pad = pad_flat(flat_bibc, MAX_BRANCHES * MAX_NON_SLACK)
            bcbv_pad = pad_flat(flat_bcbv, MAX_NON_SLACK * MAX_BRANCHES)
            wrt_target_test.writerow([system_id, topo_id] + bibc_pad + bcbv_pad)

            for prof_id in range(1, n_profiles+1):
                loads_pu = random_load_profile(config['loads'], config['base_kva'])
                V_true, I_true = run_loadflow(branches, loads_pu, config['num_buses'], config['base_kva'])
                if V_true is None:
                    continue
                V_noisy = add_noise(V_true, 1.0)
                I_noisy = add_noise(I_true, 1.0)
                V_pad = list(V_noisy) + [0.0] * (MAX_BUSES - config['num_buses'])
                I_pad = list(I_noisy) + [0.0] * (MAX_BRANCHES - len(branches))
                wrt_vi_test.writerow([system_id, topo_id, prof_id] + V_pad + I_pad)

        print(f"  Appended data for system {system_id}.")

    vi_train.close(); vi_test.close()
    target_train.close(); target_test.close()
    print("Done. Files: vi_train.csv, vi_test.csv, bibc_bcbv_train.csv, bibc_bcbv_test.csv")

if __name__ == "__main__":
    main()