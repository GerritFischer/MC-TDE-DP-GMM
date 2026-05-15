from nds_toolbox.models.dpgmm_numpyro import fit_DPGMM, truncate, get_states
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

import pickle
import os

from nds_toolbox.sim.bursts.simulator import simulate_bursty_signal
from nds_toolbox.preprocessing.features import (compute_tde,trim_data)
from nds_toolbox.preprocessing.features import choose_embedding_dim
from nds_toolbox.utils.helper import compare_decoding_performance

path = os.path.join("C:", "meshes", "as")

sim_cond = "ind10-(15-40)-snr(-10-10)"
signal_dir = os.path.join("..", "data", "simulations")
performance_dir = os.path.join("..", "data", "performance")
result_dir = "../data/results"

sim_data = np.load(f'{signal_dir}/{sim_cond}_data.npz', allow_pickle=True)
signal_sample = sim_data['signal_samples']
states_sample = sim_data['states_samples']
bursts_sample = sim_data['bursts_samples']
print("shape of signal_sample", signal_sample.shape)

seed = 2026

num_models = 5 #default: 10
num_epochs = 3000#default: 3000
n_jobs = 10 #increase the number of jobs when you want to multi process the inference
lr = 0.01
num_emb = 15  

results = np.empty((len(signal_sample), len(signal_sample[0]), len(signal_sample[0,0]), len(signal_sample[0,0,0]), len(signal_sample[0,0,0,0])-1), dtype=object)


for s_id, sample in enumerate(signal_sample):
    for cond1_id, cond1 in enumerate(sample):
        for cond2_id, cond2 in enumerate(cond1):
            for snr_id, snr in enumerate(cond2):
                main_sig = snr[0]
                main_states = states_sample[s_id, cond1_id, cond2_id, snr_id, 0]
                
                main_sig = (main_sig - np.mean(main_sig))/ np.std(main_sig)
                tde_signal = compute_tde(main_sig, num_emb)

                trimmed_main_sig = trim_data(main_sig, num_emb, verbose = False)

                comp_sig = snr[1:]
                for sig_id, sig in enumerate(comp_sig):                    
                    sig = (sig - np.mean(sig))/ np.std(sig)
                    states = states_sample[s_id, cond1_id, cond2_id, snr_id, sig_id+1]
                    trimmed_sig = trim_data(sig, num_emb, verbose= False)
                    tde_signal2 = compute_tde(sig, num_emb)                   
                    combined_tde = np.concatenate([tde_signal, tde_signal2], axis=1)

                    num_states = int(np.ceil(np.log(len(trimmed_sig)))) # E[K] = alpha ln n
                    
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
                    
                    results[s_id, cond1_id, cond2_id, snr_id, sig_id] = dict(dpgmm_result)

os.makedirs(result_dir, exist_ok = True)                    
data_file = f"{result_dir}/{sim_cond}_results.npz"
np.savez_compressed(data_file,
                    results=results)

print(f"Data saved as {data_file}")