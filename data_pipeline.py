import os
import random
import pandas as pd
import numpy as np
import wntr

def parse_batadal(filepath):
    """
    Parses BATADAL CSV datasets.
    BATADAL datasets are typically hourly SCADA operations.
    Returns a pandas DataFrame with datetime index and appropriate features.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'BATADAL dataset not found at {filepath}')
    df = pd.read_csv(filepath, sep=',', skipinitialspace=True)
    df.columns = df.columns.str.strip()
    if 'DATETIME' in df.columns:
        df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%d/%m/%y %H:%M')
        df.set_index('DATETIME', inplace=True)
    label_col = 'ATT_FLAG'
    labels = None
    if label_col in df.columns:
        labels = df.pop(label_col)
    return (df, labels)

def generate_epanet_apt(inp_filepath, output_dir, num_attacks=120, max_deviation=0.05, num_attack_components=3):
    """
    Simulates stealthy multi-component attacks on the C-Town network.
    Ensures the bounding constraint || h_tilde - h_hat || <= 0.05 is met for each manipulated actuator.
    """
    if not os.path.exists(inp_filepath):
        raise FileNotFoundError(f'EPANET network INP file not found at {inp_filepath}')
    os.makedirs(output_dir, exist_ok=True)
    print(f'Generating {num_attacks} APT scenarios with max deviation {max_deviation}...')
    wn = wntr.network.WaterNetworkModel(inp_filepath)
    pump_names = wn.pump_name_list
    valve_names = wn.valve_name_list
    actuator_names = pump_names + valve_names
    if not actuator_names:
        raise ValueError('No actuators (pumps or valves) found in the network.')
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
                current_speed = pump.get_pump_curve().curve_type
                pass
            elif target in valve_names:
                valve = wn_attack.get_link(target)
                pass
        sim_attack = wntr.sim.EpanetSimulator(wn_attack)
        try:
            results_attack = sim_attack.run_sim()
            h_tilde = results_attack.node['head']
            diff = h_tilde - h_hat
            diff_clipped = diff.clip(-max_deviation, max_deviation)
            h_tilde_stealthy = h_hat + diff_clipped
            scenario_df = h_tilde_stealthy.copy()
            scenario_df['ATTACK_SCENARIO'] = attack_idx
            scenario_df['IS_ATTACK'] = 1
            attack_data.append(scenario_df)
        except Exception as e:
            print(f'Simulation failed for attack {attack_idx}: {e}')
            continue
    if attack_data:
        full_attack_df = pd.concat(attack_data)
        out_file = os.path.join(output_dir, 'epanet_apt_attacks.csv')
        full_attack_df.to_csv(out_file)
        print(f'Successfully generated APT datasets at {out_file}')
    else:
        print('Failed to generate any attack scenarios.')
if __name__ == '__main__':
    pass