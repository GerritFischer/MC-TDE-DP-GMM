import numpy as np
from math import ceil 
from nds_toolbox.sim.bursts.simulator import (_add_noise)

def phase_shift(signal_dict, degree, snr_db, fs, freq, use_filter=True, highpass_f=0.5):
    """
    Shifts the phase of bursts by a certain degree. Adds the existing noise back on after shift.
    Does not shift the noise.
    """
    states = signal_dict["states"]
    bursts = signal_dict["unscaled_bursts"]
    noise = signal_dict["noise"]


    onset_start = 0
    onset_found = False
    for index, state in enumerate(states):
        if state != 0:
            if not onset_found:
                onset_start = index
                onset_found = True
        else:
            if onset_found:
                burst = np.copy(bursts[onset_start:index-1])
                roll_value = (degree % 360) * 1 // (360 // (fs / freq[state-1]))
                burst = np.roll(burst, roll_value)
                 

                bursts[onset_start:index-1] = np.copy(burst)
                print("phase shifted signal by " + str(roll_value))

                onset_found = False
    


    signal, scaled_bursts = _add_noise(bursts, states, noise, snr_db, use_filter = use_filter, fs = fs, highpass_f = highpass_f)


    return {"signal": signal,
            "states": states,
            "bursts": scaled_bursts,
            "unscaled_bursts": bursts,
            "noise": noise,}


def delay_bursts(signal_dict, snr_db, delay, fs, use_filter=True, highpass_f=0.5):
    """
    Delays the bursts in a existing signal by a certain amount of time given in ms
    """



    states = np.copy(signal_dict["states"])
    bursts = np.copy(signal_dict["unscaled_bursts"])
    noise = np.copy(signal_dict["noise"])
    
    offset = ceil(delay * (fs / 1000))

    states = np.roll(states, offset)
    bursts = np.roll(bursts, offset)
    
    print(f"shifted by {offset}")
    signal, scaled_bursts = _add_noise(bursts, states, noise, snr_db, use_filter = use_filter, fs = fs, highpass_f = highpass_f)

    return {"signal": signal,
            "states": states,
            "bursts": scaled_bursts,
            "unscaled_bursts": bursts,
            "noise": noise,}