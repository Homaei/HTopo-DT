import os
import random
import pandas as pd
import numpy as np
import wntr
import networkx as nx

def parse_batadal(filepath):
    """
    Parses BATADAL CSV datasets.
    BATADAL datasets are typically hourly SCADA operations.
    Returns a pandas DataFrame with datetime index and appropriate features.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"BATADAL dataset not found at {filepath}")
    
    df = pd.read_csv(filepath, sep=',', skipinitialspace=True)
    df.columns = df.columns.str.strip()
    
    if 'DATETIME' in df.columns:
        df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%d/%m/%y %H:%M')
        df.set_index('DATETIME', inplace=True)
        
    label_col = 'ATT_FLAG'
    labels = None
    if label_col in df.columns:
        labels = df.pop(label_col)
        
    return df, labels

def pairwise_anomaly_check(h_history, h_new, window_size=288, threshold=3.0):
    """
    Performs a standard per-component pairwise anomaly check using a rolling z-score.
    h_history: array of shape (window_size, num_components)
    h_new: array of shape (num_components,)
    Returns True if an anomaly is detected on ANY component, False otherwise.
    """
    if len(h_history) < window_size:
        return False
        
    mu = np.mean(h_history[-window_size:], axis=0)
    std = np.std(h_history[-window_size:], axis=0)
    
    # Avoid division by zero
    std = np.where(std < 1e-6, 1e-6, std)
    
    z_scores = np.abs((h_new - mu) / std)
    
    # If any component exceeds the threshold, it is not stealthy
    if np.any(z_scores > threshold):
        return True
    return False

def verify_meshed_topology(wn):
    """
    Verifies that the network is meshed and has at least one independent hydraulic loop.
    """
    G = wn.get_graph().to_undirected()
    cycles = nx.cycle_basis(G)
    
    num_loops = len(cycles)
    if num_loops == 0:
        raise ValueError("Network has a sequential/tree topology with 0 independent loops. "
                         "HTopo-DT requires meshed networks (e.g., C-Town has 17 loops).")
    
    print(f"Topology Verification Passed: Found {num_loops} independent hydraulic loops.")
    return cycles

def generate_epanet_apt(inp_filepath, output_dir, num_attacks=120, max_deviation=0.05, num_attack_components=3):
    """
    Simulates stealthy multi-component attacks on the C-Town network.
    Enforces the bounding constraint || h_tilde_ij(t) - h_hat_ij(t) || <= 0.05 per component independently.
    """
    if not os.path.exists(inp_filepath):
        raise FileNotFoundError(f"EPANET network INP file not found at {inp_filepath}")
        
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_attacks} APT scenarios with max deviation {max_deviation}...")
    
    wn = wntr.network.WaterNetworkModel(inp_filepath)
    
    # Verify topology requirement (must be meshed)
    verify_meshed_topology(wn)
    
    pump_names = wn.pump_name_list
    valve_names = wn.valve_name_list
    actuator_names = pump_names + valve_names
    
    if not actuator_names:
        raise ValueError("No actuators (pumps or valves) found in the network.")
        
    wn.options.time.duration = 24 * 3600
    sim_nominal = wntr.sim.EpanetSimulator(wn)
    results_nominal = sim_nominal.run_sim()
    
    h_hat = results_nominal.node['head']
    
    attack_data = []
    
    for attack_idx in range(num_attacks):
        wn_attack = wntr.network.WaterNetworkModel(inp_filepath)
        wn_attack.options.time.duration = 24 * 3600
        
        k = min(num_attack_components, len(actuator_names))
        targets = random.sample(actuator_names, k)
        
        for target in targets:
            if target in pump_names:
                pump = wn_attack.get_link(target)
                pump.base_speed = pump.base_speed * random.uniform(1.1, 1.4)
            elif target in valve_names:
                valve = wn_attack.get_link(target)
                valve.initial_setting = valve.initial_setting * random.uniform(0.5, 0.8)
                
        sim_attack = wntr.sim.EpanetSimulator(wn_attack)
        try:
            results_attack = sim_attack.run_sim()
            h_tilde = results_attack.node['head']
            
            # 1. Enforce stealthiness per component independently
            diff = h_tilde - h_hat
            diff_clipped = diff.clip(-max_deviation, max_deviation)
            h_tilde_stealthy = h_hat + diff_clipped
            
            # 2. Pairwise anomaly check using rolling z-score (M=288)
            # We mock a continuous simulation by taking the generated heads as the new state
            # and comparing it against a history buffer (here, we use h_hat as historical baseline)
            
            # Since h_hat contains multiple time steps (e.g. 24 hours at 1 hr interval = 25 steps),
            # we simulate an online check over the time series
            
            stealthy = True
            for t in range(len(h_tilde_stealthy)):
                if t < 2:  # Need some history to compute variance
                    continue
                    
                # Use history up to t-1
                h_history = h_hat.iloc[:t].values
                h_new = h_tilde_stealthy.iloc[t].values
                
                # We use window_size up to 288 (here limited by simulation duration)
                if pairwise_anomaly_check(h_history, h_new, window_size=min(t, 288), threshold=3.0):
                    stealthy = False
                    break
            
            if stealthy:
                scenario_df = h_tilde_stealthy.copy()
                scenario_df['ATTACK_SCENARIO'] = attack_idx
                scenario_df['IS_ATTACK'] = 1
                attack_data.append(scenario_df)
            else:
                print(f"Attack {attack_idx} rejected: Pairwise anomaly check triggered an alarm.")
            
        except Exception as e:
            print(f"Simulation failed for attack {attack_idx}: {e}")
            continue
            
    if attack_data:
        full_attack_df = pd.concat(attack_data)
        out_file = os.path.join(output_dir, "epanet_apt_attacks.csv")
        full_attack_df.to_csv(out_file)
        print(f"Successfully generated APT datasets at {out_file}")
    else:
        print("Failed to generate any stealthy attack scenarios.")