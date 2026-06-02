import numpy as np
import os

simulation_condition = "testing_config"





seed = 2026

n_samples = 3 # amount of samples

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
generation_type = [0, 0, 0, 0, 0, 0, 0]  
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


phase_shift_degree = [0, 0, 0, 0, 0, 0, 0]
delays = [0, 0, 0, 0, 0, 0, 0] # delay in ms                           


##### Conditions #####

# Condition 1: Frequency range (Hz)
freq_ranges = [
                [[10], [15], [20], [25], [30], [35], [40]] #first freq range with frist signal containing 10hz osc and second signal containing 20hz osc 
]

# Condition 2: Sampling frequency [fs]
fs_range = [250]

# Signal to noise ratios
snrs = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10] #signal to noise ratios


num_models = 5 #default: 10
num_epochs = 3000#default: 3000

lr = 0.01

num_emb = 21







### generating file paths
file_name = f"{simulation_condition}_config.npz"
config_dir = os.path.join("..", "data", "configs")
file_path = os.path.join(config_dir, file_name)

os.makedirs(config_dir, exist_ok = True)


global_parameters = dict(seed=seed, n_samples=n_samples, n_seconds=n_seconds, burst_amp_sigma=burst_amp_sigma,
                        beta=beta, burst_cycles=burst_cycles, noise_duration=noise_duration, randomize_amps=randomize_amps)



fitting_parameters = dict(num_models=num_models, num_epochs=num_epochs, lr=lr, num_emb=num_emb)


np.savez_compressed(file_path,
                    global_parameters=global_parameters,
                    generation_type=generation_type,
                    phase_shift_degree=phase_shift_degree,
                    delays=delays,
                    freq_ranges=freq_ranges,
                    fs_range=fs_range,
                    snrs=snrs,
                    fitting_parameters=fitting_parameters

)
print(f"config file saved as {file_path}")

