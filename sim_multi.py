import numpy as np
import os
from nds_toolbox.sim.bursts.simulator import (simulate_bursty_signal, 
                                             simulate_empty_signal, 
                                             simulate_noise, 
                                             simulate_independent_signal, 
                                             phase_shift,
                                             copy_signal_change_burst_freq,
                                             copy_signal_change_noise,
                                             delay_bursts)
import matplotlib.pyplot as plt
import math

###################
#### SETTINGS #####
###################

use_config_file = False  #if true settings will be ignored and config will be loaded

simulation_condition = "test"

# Set seeds for reproducibility.
seed = 2026

##### Channel effect (WIP) #####
#delay_per_mm = 1000 # 1ms per mm
#decay_per_mm = 0.1 # the signal decay by 10% per mm

#enable_channel_effect = 0


##### Simulation parameters #####
n_samples = 1 # amount of samples

n_seconds = 180 # total duration in seconds.

burst_amp_sigma = 0.1
beta = 1 #pink noise

# For burst segments, specify duration as the number of cycles.
burst_cycles = [3, 7]

# For noise segments, specify duration in seconds.
noise_duration = [0.5, 3.]


# Used for copy onset times with new frequency, specify if amps 
# should be randomized during a single onset
randomize_amps = True

##### Settings per channel ######

### the length of this array determines how many channels are created
### make sure to have at least that many freqs, if more freqs are present they are ignored
generation_type = [0, 0]
                           #### new signals ####
                           #0 = new signal from scratch
                           #1 = empty signal
                           #2 = only noise
                           #3 = independent onsets
                           #-------------------------------#
                           #### copied signals ####
                           #4 = copy completly
                           #5 = copy onset times
                           #6 = copy bursts


phase_shift_degree = [0, 0, 0, 0, 0, 0, 0, 0]
delays = [0, 0, 0, 0, 0, 0, 0, 0] # delay in ms                           


##### Conditions #####

# Condition 1: Frequency range (Hz)
freq_ranges = [
                [[20], [5], [10], [15], [25], [30], [35], [40]] #first freq range with frist signal containing 10hz osc and second signal containing 20hz osc 
]

# Condition 2: Sampling frequency [fs]
fs_range = [250]

# Signal to noise ratios
#snrs = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10] #signal to noise ratios
snrs = [2]
###################################################################



rng = np.random.default_rng(seed)


### generating file paths
file_name = f"{simulation_condition}_data.npz"
figure_dir = os.path.join("..", "figures", simulation_condition)
data_dir = os.path.join("..", "data", "simulations")
file_path = os.path.join(data_dir, file_name)

(os.makedirs
 (figure_dir, exist_ok=True))
os.makedirs(data_dir, exist_ok = True)




# create arrays based on parameters above
states_samples = np.empty((n_samples, len(fs_range), len(fs_range), len(snrs), len(generation_type)), dtype=object)
bursts_samples = np.empty_like(states_samples)
noise_samples  = np.empty_like(states_samples)
signal_samples = np.empty_like(states_samples)
unscaled_bursts_samples = np.empty_like(states_samples)

### generate channels for each sample, condition and snr
for sample_id in range(n_samples):
    for cond1_id, freq in enumerate(freq_ranges):
        for cond2_id, fs in enumerate(fs_range):
            for snr_id, snr in enumerate(snrs):
                for signal_pos_id, signal in enumerate(generation_type):

                    time_vec = np.linspace(0, n_seconds, int(fs * n_seconds), endpoint=False)
                    gtype = generation_type[signal_pos_id]


                    if gtype == 1: #empty channel
                        signal_dict = simulate_empty_signal(time_vec)
                    elif gtype == 2: #noise only
                        signal_dict = simulate_noise(time_vec, fs, beta, rng)
                    elif gtype == 3:
                        signal_dict = simulate_independent_signal(
                            time_vec, fs, freq[signal_pos_id], burst_cycles, noise_duration,
                            burst_type="sine", snr_db=snr, beta=beta,
                            burst_amp_sigma=burst_amp_sigma, rng=rng, prev_states=states_samples[sample_id, cond1_id, cond2_id, snr_id, :1]
                            )                                                                                                       #set to :signal_pos_id to make each channel independent form all
                                                                                                                                    #set to :1 to make channel 1 to N independent from first
                    elif gtype == 4:  #copy complete
                        signal_dict = {"signal": np.copy(signal_samples[sample_id, cond1_id, cond2_id, snr_id, 0]),
                                    "states": np.copy(states_samples[sample_id, cond1_id, cond2_id, snr_id, 0]),
                                    "bursts": np.copy(bursts_samples[sample_id, cond1_id, cond2_id, snr_id, 0]),
                                    "unscaled_bursts": np.copy(unscaled_bursts_samples[sample_id, cond1_id, cond2_id, snr_id, 0]),
                                    "noise": np.copy(noise_samples[sample_id, cond1_id, cond2_id, snr_id, 0]),}
                    elif gtype == 5: #copy onset times with changed freq
                        signal_dict = {"signal": np.copy(signal_samples[sample_id, cond1_id, cond2_id, snr_id, 0]),
                                    "states": np.copy(states_samples[sample_id, cond1_id, cond2_id, snr_id, 0]),
                                    "bursts": np.copy(bursts_samples[sample_id, cond1_id, cond2_id, snr_id,0]),
                                    "unscaled_bursts": np.copy(unscaled_bursts_samples[sample_id, cond1_id, cond2_id, snr_id, 0]),
                                    "noise": np.copy(noise_samples[sample_id, cond1_id, cond2_id, snr_id, 0]),}
                        signal_dict = copy_signal_change_burst_freq(signal_dict, snr, fs, freq[signal_pos_id], rng, time_vec, scale_amps=randomize_amps)

                    elif gtype == 6: ##copy bursts with new noise
                        signal_dict = {"signal": np.copy(signal_samples[sample_id, cond1_id, cond2_id, snr_id, 0]),
                                    "states": np.copy(states_samples[sample_id, cond1_id, cond2_id, snr_id, 0]),
                                    "bursts": np.copy(bursts_samples[sample_id, cond1_id, cond2_id, snr_id, 0]),
                                    "unscaled_bursts": np.copy(unscaled_bursts_samples[sample_id, cond1_id, cond2_id, snr_id, 0]),
                                    "noise": np.copy(noise_samples[sample_id, cond1_id, cond2_id, snr_id, 0]),}
                        signal_dict = copy_signal_change_noise(signal_dict, snr, fs, rng, time_vec)

                    else: #normal signal 
                        signal_dict = simulate_bursty_signal(
                            time_vec, fs, freq[signal_pos_id], burst_cycles, noise_duration,
                            burst_type="sine", snr_db=snr, beta=beta,
                            burst_amp_sigma=burst_amp_sigma, rng=rng
                        )

                    
                    #apply phase shift if specified in dict 
                    if phase_shift_degree[signal_pos_id] != 0:
                        signal_dict = phase_shift(signal_dict, phase_shift_degree[signal_pos_id], snr, fs, freq[signal_pos_id])
                    
                    if delays[signal_pos_id] != 0:
                        signal_dict = delay_bursts(signal_dict, snr, delays[signal_pos_id], fs) 

                    states_samples[sample_id, cond1_id, cond2_id, snr_id, signal_pos_id]  = signal_dict["states"]
                    bursts_samples[sample_id, cond1_id, cond2_id, snr_id, signal_pos_id]  = signal_dict["bursts"]
                    noise_samples [sample_id, cond1_id, cond2_id, snr_id, signal_pos_id]  = signal_dict["noise"]
                    signal_samples[sample_id, cond1_id, cond2_id, snr_id, signal_pos_id]  = signal_dict["signal"]
                    unscaled_bursts_samples[sample_id, cond1_id, cond2_id, snr_id, signal_pos_id] = signal_dict["unscaled_bursts"]



print("Data shape", signal_samples.shape, "[samples, cond1, cond2, snrs, signals]: the data points are stored as object")




### calculate influence of signals to each other (WIP)
'''
final_signal_samples = np.empty_like(signal_samples)


for sample_id,sample in enumerate(signal_samples):
    for sig_id, signal in enumerate(sample):
        new_sig = np.copy(signal)
        sig_pos = signal_positions[sig_id] 
        for cond1_id in range(len(freq_range)):
            for cond2_id in range(len(fs_range)):
                for app_sig_id, app_signal in enumerate(signal_samples):
                    if empty_signal[app_sig_id] or sig_id == app_sig_id: 
                        continue
                    app_pos = signal_positions[app_sig_id]
                    distance = math.sqrt((sig_pos[0] - app_pos[0]) ** 2 + (sig_pos[1] - app_pos[1]) ** 2 + (sig_pos[2] - app_pos[2]) ** 2)

                    ##todo: add support for multiple samples and freqs

                    offset = math.ceil((fs_range[cond2_id] / 1000) * (distance * delay_per_mm))
                    decay = (1- decay_per_mm) ** distance 
                    print(distance)
                    print(decay)
                    act_sig = new_sig[sample_id,cond1_id,cond2_id]
                    act_app_sig = app_signal[sample_id,cond1_id,cond2_id]

                    for s, sig in enumerate(act_app_sig):
                        if (s+offset) < len(act_sig):
                            act_sig[s+offset] += act_app_sig[s] * decay

                    new_sig[sample_id,cond1_id,cond2_id] = act_sig

                final_signal_samples[sig_id] = new_sig
        
'''


### save data
np.savez_compressed(file_path,
                    signal_samples=signal_samples,
                    states_samples=states_samples,
                    bursts_samples=bursts_samples,
                    noise_samples=noise_samples)

print(f"Data saved as {file_path}")



### DEBUG CODE
"""
fig, axes = plt.subplots(3, 2)


axes[0,0].set_title('Original Signal')
axes[0,0].plot(unscaled_bursts_samples[0,0,0,0,0])


axes[0,1].set_title('Independent Onset')
axes[0,1].plot(unscaled_bursts_samples[0,0,0,0,0])
axes[0,1].plot(unscaled_bursts_samples[0,0,0,0,1])

print(states_samples[0,0,0,0,0][11999])
print(states_samples[0,0,0,0,1][11999])

axes[1,0].set_title('Simultaneous Onset, Same Phase')
axes[1,0].plot(unscaled_bursts_samples[0,0,0,0])
axes[1,0].plot(unscaled_bursts_samples[0,0,0,2])


axes[1,1].set_title('Simultaneous Onset, Different Phase')
axes[1,1].plot(unscaled_bursts_samples[0,0,0,0])
axes[1,1].plot(unscaled_bursts_samples[0,0,0,3])


axes[2,0].set_title('Simultaneous Onset, Different Frequency')
axes[2,0].plot(signal_samples[0,0,0,0])
axes[2,0].plot(signal_samples[0,0,0,4])

axes[2,1].set_title('\"Near Simultaneous\" Onset, 100ms Delay')
axes[2,1].plot(unscaled_bursts_samples[0,0,0,0])
axes[2,1].plot(unscaled_bursts_samples[0,0,0,5])

plt.show()         
"""
