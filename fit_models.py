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

sim_cond = "independent-40-[10:35]-snr-[-10:10]"
signal_dir = "../data/simulations"
performance_dir = "../data/performance"
result_dir = "../data/results"

sim_data = np.load(f'{signal_dir}/{sim_cond}_data.npz', allow_pickle=True)
signal_sample = sim_data['signal_samples']
states_sample = sim_data['states_samples']
bursts_sample = sim_data['bursts_samples']
print("shape of signal_sample", signal_sample.shape)

freq = np.array([10])
freq2 = np.array([20])
seed = 2026
fs = 200
time_vec = np.linspace(0, 60, int(fs * 60))
snr_db = 2
state_transition = 'return_to_baseline'
num_emb = 21
results = np.empty_like(signal_sample)

num_models = 1 #default: 10
num_epochs = 3000#default: 3000
n_jobs = 1 #increase the number of jobs when you want to multi process the inference
lr = 1e-2


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


                    combined_states = np.empty_like(states)
                        
                    state_dict = {'00': 0}
                    state_num = 1
                    for i, (s1, s2) in enumerate(zip(main_states, states)):
                        s =  str(s1) + str(s2)
                        if s not in state_dict:
                            state_dict[s] = state_num
                            state_num += 1
                        combined_states[i] = state_dict[s]

                    trimmed_states = trim_data(combined_states, num_emb, verbose = False)
                    trimmed_time_vec = np.linspace(0, len(trimmed_states)/fs, len(trimmed_states))
                    num_states = int(np.ceil(np.log(len(trimmed_sig)))) # E[K] = alpha ln n
                    
                    dpgmm_result = fit_DPGMM(data=combined_tde,
                                            num_states=num_states,
                                            use_epoch_tqdm=True,
                                            n_jobs=10,
                                            use_model_tqdm=True,
                                            num_models=num_models,
                                            main_seed=seed,
                                            num_epochs = num_epochs,
                                            learning_rate=lr,
                                            verbose=True)
                    
                    results[s_id, cond1_id, cond2_id, snr_id, sig_id] = np.copy(dpgmm_result)

os.makedirs(result_dir, exist_ok = True)                    
data_file = f"{result_dir}/{sim_cond}_results.npz"
np.savez_compressed(data_file,
                    results=results)

print(f"Data saved as {data_file}")