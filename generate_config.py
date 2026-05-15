rng = 2026

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











