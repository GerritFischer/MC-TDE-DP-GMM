import os
import numpy as np


"""
This is a script for combining two datasets and results together.
They have to be split in multiple chunks of SNRs.
For example you can have two sets like this:
    (3, 1, 1, 5, 10)
    (3, 1, 1, 7, 10)

with 3 samples, 1 condition_1, 1 condition_2, 5/7 different snrs and 10 channels.
They will be combined into a single dataset and result file in the format of:
    (3, 1, 1, 12, 10)

This script was created to fit the models for larger datasets on different computers.
"""




con1 = "rnd-5-(10-40)-snr(-10-0)"
con2 = "rnd-5-(10-40)-snr(2-10)"
con_fin = "rnd-5-(10-40)-snr(-10-10)"


result_dir = os.path.join ("..", "data", "results")
sim_dir = os.path.join("..", "data", "simulations")

res1 = os.path.join(result_dir, f"{con1}_results.npz")
res2 = os.path.join(result_dir, f"{con2}_results.npz")
res_fin = os.path.join(result_dir, f"{con_fin}_results.npz")

sim1 = os.path.join(sim_dir, f"{con1}_data.npz")
sim2 = os.path.join(sim_dir, f"{con2}_data.npz")
sim_fin = os.path.join(sim_dir, f"{con_fin}_data.npz")

res1_data = np.load(res1, allow_pickle=True)
res2_data = np.load(res2, allow_pickle=True)

sim1_data = np.load(sim1, allow_pickle=True)
sim2_data = np.load(sim2, allow_pickle=True)

res1_array = res1_data['results']
res2_array = res2_data['results']



signal_sample1 = sim1_data['signal_samples']
states_sample1 = sim1_data['states_samples']
bursts_sample1 = sim1_data['bursts_samples']
noise_sample1 = sim1_data['noise_samples']

signal_sample2 = sim2_data['signal_samples']
states_sample2 = sim2_data['states_samples']
bursts_sample2 = sim2_data['bursts_samples']
noise_sample2 = sim2_data['noise_samples']

res1_array = res1_data['results']
res2_array = res2_data['results']


print(f"RESULT 1 SHAPE {res1_array.shape}")

print(f"RESULT 2 SHAPE {res2_array.shape}")

res_fin_array = np.concat((res1_array, res2_array), axis=3)

signal_sample_fin = np.concat((signal_sample1, signal_sample2), axis=3)
states_sample_fin = np.concat((states_sample1, states_sample2), axis=3)
bursts_sample_fin = np.concat((bursts_sample1, bursts_sample2), axis=3)
noise_sample_fin = np.concat((noise_sample1, noise_sample2), axis=3)


print(f"RESULT FINAL SHAPE {res_fin_array.shape}")


np.savez_compressed(res_fin,
                    results=res_fin_array)

np.savez_compressed(sim_fin,
                    signal_samples=signal_sample_fin,
                    states_samples=states_sample_fin,
                    bursts_samples=bursts_sample_fin,
                    noise_samples=noise_sample_fin)
