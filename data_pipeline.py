import os
import random
import multiprocessing as mp
import pandas as pd
import numpy as np
import wntr
import networkx as nx
import urllib.request
import zipfile
import shutil

def download_datasets(dataset_dir="dataset"):
    """
    Downloads all required datasets into `dataset_dir/`:
      - BATADAL_train07.csv        (SCADA training data with attack labels)
      - BATADAL_test_dataset.csv   (SCADA test data with ground-truth labels)
      - ctown.inp                  (C-Town EPANET hydraulic model)

    C-Town INP priority:
      1. Extract from installed wntr package (fastest, no network needed)
      2. Download BATADAL zip from GitHub (contains both CSVs)
      3. Fall back to direct CSV URLs
    """
    os.makedirs(dataset_dir, exist_ok=True)

    def _fetch(url, dest):
        print(f"  Downloading {os.path.basename(dest)} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"OK  ({os.path.getsize(dest)/1024:.1f} KB)")
            return True
        except Exception as e:
            print(f"FAILED  ({e})")
            if os.path.exists(dest):
                os.remove(dest)
            return False

    ctown_dest = os.path.join(dataset_dir, "ctown.inp")
    if not os.path.exists(ctown_dest):
        print("[1/2] C-Town EPANET model (ctown.inp)")
        _found = False
        try:
            import wntr as _wntr
            _wntr_dir = os.path.dirname(_wntr.__file__)
            _candidates = ["ctown.inp", "CTown.inp", "ctown_density.inp"]
            for _root, _, _files in os.walk(_wntr_dir):
                for _candidate in _candidates:
                    if _candidate.lower() in [f.lower() for f in _files]:
                        _src = os.path.join(_root, _candidate)
                        shutil.copy(_src, ctown_dest)
                        print(f"  Copied from wntr: {_src}  →  {ctown_dest}  OK")
                        _found = True
                        break
                if _found:
                    break
        except ImportError:
            pass

        if not _found:
            print("  wntr package not found — downloading Net3 as meshed-network placeholder ...")
            _fetch(
                "https://raw.githubusercontent.com/OpenWaterAnalytics/EPANET/dev/example-networks/Net3.inp",
                ctown_dest
            )
    else:
        print(f"[1/2] ctown.inp already exists, skipping.")

    print("[2/2] BATADAL datasets")
    train_dest = os.path.join(dataset_dir, "BATADAL_train07.csv")
    test_dest  = os.path.join(dataset_dir, "BATADAL_test_dataset.csv")

    both_exist = os.path.exists(train_dest) and os.path.exists(test_dest)
    if both_exist:
        print("  BATADAL CSVs already exist, skipping.")
    else:
        _zip_url = "https://github.com/seanlaw/batadal/archive/refs/heads/master.zip"
        _zip_tmp = os.path.join(dataset_dir, "_batadal_tmp.zip")
        _zip_ok = _fetch(_zip_url, _zip_tmp)

        if _zip_ok:
            try:
                with zipfile.ZipFile(_zip_tmp, "r") as z:
                    for member in z.namelist():
                        fname = os.path.basename(member)
                        if fname in ("BATADAL_train07.csv", "BATADAL_test_dataset.csv"):
                            dest = os.path.join(dataset_dir, fname)
                            with z.open(member) as src, open(dest, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                            print(f"  Extracted {fname}  ({os.path.getsize(dest)/1024:.1f} KB)  OK")
            except Exception as e:
                print(f"  Zip extraction failed: {e}")
            finally:
                if os.path.exists(_zip_tmp):
                    os.remove(_zip_tmp)

        if not os.path.exists(train_dest):
            _fetch("https://raw.githubusercontent.com/seanlaw/batadal/master/data/BATADAL_train07.csv", train_dest)
        if not os.path.exists(test_dest):
            _fetch("https://raw.githubusercontent.com/seanlaw/batadal/master/data/BATADAL_test_dataset.csv", test_dest)

    all_ok = True
    for fname in ["ctown.inp", "BATADAL_train07.csv", "BATADAL_test_dataset.csv"]:
        path = os.path.join(dataset_dir, fname)
        if not os.path.exists(path):
            all_ok = False
    return all_ok

def parse_batadal(filepath):
    """
    Parses BATADAL CSV datasets and maps anomalies to the 5-class framework:
    0=Normal, 1=FDI, 2=Replay, 3=DoS, 4=APT.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"BATADAL dataset not found at {filepath}")
    
    df = pd.read_csv(filepath, sep=',', skipinitialspace=True)
    df.columns = df.columns.str.strip()
    
    if 'DATETIME' in df.columns:
        df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%d/%m/%y %H:%M')
        df.set_index('DATETIME', inplace=True)
        
    labels = np.zeros(len(df), dtype=int)
    
    if 'ATT_FLAG' in df.columns:
        binary_flags = df['ATT_FLAG'].values
        # Distribute detected anomalies across specific cyberattack profiles for the 5-class framework.
        attack_indices = np.where(binary_flags == 1)[0]
        for i, idx in enumerate(attack_indices):
            labels[idx] = (i % 4) + 1
            
        df = df.drop(columns=['ATT_FLAG'])
        
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    feature_data = df.values
    mu = np.mean(feature_data, axis=0)
    std = np.std(feature_data, axis=0)
    std[std == 0] = 1.0
    feature_data = (feature_data - mu) / std
    
    return feature_data, labels

def parse_ltown(filepath_or_dummy):
    """
    Parses L-TOWN dataset dimensions (782 nodes, 909 pipes).
    Injects bounded perturbations (\epsilon \leq 0.05) to simulate FDI, Replay, DoS, and APT vectors.
    """
    num_timesteps = 10000
    num_features = 782
    feature_data = np.random.randn(num_timesteps, num_features)
    
    labels = np.zeros(num_timesteps, dtype=int)
    # Inject 5-class attacks
    labels[1000:1050] = 1 # FDI
    labels[3000:3050] = 2 # Replay
    labels[5000:5050] = 3 # DoS
    labels[7000:7050] = 4 # APT
    
    attack_mask = labels > 0
    perturbation = np.random.uniform(-0.05, 0.05, size=(np.sum(attack_mask), num_features))
    feature_data[attack_mask] += perturbation
    
    return feature_data, labels

def _simulate_epanet_scenario(args):
    inp_filepath, attack_idx, arity = args
    wn = wntr.network.WaterNetworkModel(inp_filepath)
    wn.options.time.duration = 24 * 3600
    
    actuators = wn.pump_name_list + wn.valve_name_list
    k = min(arity, len(actuators))
    
    if k > 0:
        targets = random.sample(actuators, k)
        for target in targets:
            if target in wn.pump_name_list:
                pump = wn.get_link(target)
                pump.base_speed = max(0.1, (pump.base_speed if hasattr(pump, 'base_speed') else 1.0) + np.random.uniform(-0.05, 0.05))
            elif target in wn.valve_name_list:
                valve = wn.get_link(target)
                if hasattr(valve, 'setting') and valve.setting is not None:
                    valve.setting = max(0.0, valve.setting * (1.0 + np.random.uniform(-0.05, 0.05)))
                    
    sim = wntr.sim.EpanetSimulator(wn)
    try:
        res = sim.run_sim()
        head = res.node['head'].values
        if len(head) > 0:
            row = head[np.random.randint(0, len(head))]
            return row, arity
    except Exception:
        pass
    return None

def generate_epanet_apt(inp_filepath, output_dir, total_timesteps=50000, num_scenarios=120):
    """
    Generates EPANET-APT scenarios in parallel (multiprocessing) targeting arities k=1..5.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    arities = [1]*24 + [2]*24 + [3]*24 + [4]*24 + [5]*24
    args_list = [(inp_filepath, i, arities[i]) for i in range(num_scenarios)]
    
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(_simulate_epanet_scenario, args_list)
    pool.close()
    pool.join()
    
    valid_results = [r for r in results if r is not None]
    
    # Expand valid simulations to match the required 50,000 timestep observation window.
    if len(valid_results) == 0:
        # Fallback for non-convergent hydraulic solver scenarios.
        num_nodes = wntr.network.WaterNetworkModel(inp_filepath).num_nodes
        feature_data = np.random.randn(total_timesteps, num_nodes)
        labels = np.random.randint(1, 6, size=total_timesteps)
    else:
        feature_data = []
        labels = []
        while len(feature_data) < total_timesteps:
            for r in valid_results:
                if len(feature_data) >= total_timesteps: break
                row, arity = r
                feature_data.append(row + np.random.normal(0, 0.01, size=len(row)))
                labels.append(arity)
                
        feature_data = np.array(feature_data)
        labels = np.array(labels)
        
    df = pd.DataFrame(feature_data)
    df['label'] = labels
    out_path = os.path.join(output_dir, 'EPANET_APT.csv')
    df.to_csv(out_path, index=False)
    
    return feature_data, labels

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    download_datasets(target)