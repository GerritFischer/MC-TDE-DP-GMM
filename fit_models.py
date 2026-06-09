from nds_toolbox.models.dpgmm_numpyro import fit_DPGMM, truncate, get_states
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle

from nds_toolbox.sim.bursts.simulator import simulate_bursty_signal
from nds_toolbox.preprocessing.features import (compute_tde,trim_data)
from nds_toolbox.preprocessing.features import choose_embedding_dim
from nds_toolbox.utils.helper import compare_decoding_performance

config_file = None
# load config file if path was given as arg
try:
    config_path = sys.argv[1]
    config_file = np.load(config_path, allow_pickle=True)
except IndexError:
    print("No config path was given, using config defined in code")

#############
n_jobs = 10
#############

simulation_condition = "emb_test_fs250_ind"

seed = 2026

num_models = 5 #default: 10
num_epochs = 3000#default: 3000

lr = 0.01

num_emb = 21

##if this is true, a different embedding size is used per snr
num_embs = [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35] 
use_different_emb = True

############################

#overwrite config if file was given
if config_file is not None:

    simulation_condition = os.path.split(config_path)[1][:-11]

    global_params = config_file["global_parameters"].item()
    seed = global_params["seed"]

    fitting_parameters = config_file["fitting_parameters"].item()
    num_models = fitting_parameters["num_models"]
    num_epochs = fitting_parameters["num_epochs"]
    lr = fitting_parameters["lr"]
    num_emb = fitting_parameters["num_emb"]

############################



signal_dir = os.path.join("..", "data", "simulations")
performance_dir = os.path.join("..", "data", "performance")
result_dir = os.path.join ("..", "data", "results")

sim_path = os.path.join(signal_dir, f"{simulation_condition}_data.npz")

sim_data = np.load(sim_path, allow_pickle=True)
signal_sample = sim_data['signal_samples']
states_sample = sim_data['states_samples']
bursts_sample = sim_data['bursts_samples']


print(f"fitting models with condition {simulation_condition}")
print("shape of signal_sample", signal_sample.shape)

total_runs = len(signal_sample) * len(signal_sample[0]) * len(signal_sample[0,0]) * len(signal_sample[0,0,0]) * (len(signal_sample[0,0,0,0])-1)
current_run = 1

results = np.empty((len(signal_sample), len(signal_sample[0]), len(signal_sample[0,0]), len(signal_sample[0,0,0]), len(signal_sample[0,0,0,0])-1), dtype=object)



for s_id, sample in enumerate(signal_sample):
    for cond1_id, cond1 in enumerate(sample):
        for cond2_id, cond2 in enumerate(cond1):
            for snr_id, snr in enumerate(cond2):
                if use_different_emb:
                    num_emb = num_embs[snr_id]

                main_sig = snr[0]
                main_states = states_sample[s_id, cond1_id, cond2_id, snr_id, 0]
                
                main_sig = (main_sig - np.mean(main_sig))/ np.std(main_sig)
                tde_signal = compute_tde(main_sig, num_emb)

                trimmed_main_sig = trim_data(main_sig, num_emb, verbose=False)

                comp_sig = snr[1:]
                for sig_id, sig in enumerate(comp_sig):                    
                    sig = (sig - np.mean(sig))/ np.std(sig)
                    states = states_sample[s_id, cond1_id, cond2_id, snr_id, sig_id+1]
                    trimmed_sig = trim_data(sig, num_emb, verbose=False)
                    tde_signal2 = compute_tde(sig, num_emb)                   
                    combined_tde = np.concatenate([tde_signal, tde_signal2], axis=1)

                    num_states = int(np.ceil(np.log(len(trimmed_sig)))) # E[K] = alpha ln n
                    
                    print(f"Runnig: {current_run}/{total_runs}")
                    dpgmm_result = fit_DPGMM(data=combined_tde,
                                            num_states=num_states,
                                            use_epoch_tqdm=True,
                                            n_jobs=n_jobs,
                                            use_model_tqdm=True,
                                            num_models=num_models,
                                            main_seed=seed,
                                            num_epochs = num_epochs,
                                            learning_rate=lr,
                                            verbose=True)
                    
                    current_run += 1
                    results[s_id, cond1_id, cond2_id, snr_id, sig_id] = dict(dpgmm_result)

os.makedirs(result_dir, exist_ok = True)                    
result_path = os.path.join(result_dir, f"{simulation_condition}_results.npz")
np.savez_compressed(result_path,
                    results=results)

print(f"Results saved as {result_path}")
