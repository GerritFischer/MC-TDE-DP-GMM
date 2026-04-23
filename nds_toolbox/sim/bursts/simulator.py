"""
The following burst simulator is inspired from:
- "Quinn2019_BurstHMM/hmm_util_get_simulation.m"
- "Cho2022_BurstDetection/utils/util_data/generates_simulaiton.m"
- Cole, S., Donoghue, T., Gao, R., & Voytek, B. (2019). NeuroDSP: A package for
neural digital signal processing. Journal of Open Source Software, 4(36), 1272.
DOI: 10.21105/joss.01272
"""


import numpy as np
from scipy.signal import sawtooth
from scipy.signal.windows import tukey
from neurodsp.filt import filter_signal
from math import ceil 
import matplotlib.pyplot as plt



def _get_duration(duration_param, rng):
    """
    Determines a duration value.

    If duration_param is a scalar (int or float), returns it.
    If it is a two-element list/tuple/array, returns a random value drawn uniformly
    between the two values.
    """
    if isinstance(duration_param, (list, tuple, np.ndarray)) and len(duration_param) == 2:
        return rng.uniform(duration_param[0], duration_param[1])
    elif np.isscalar(duration_param):
        return duration_param
    else:
        raise ValueError("Duration parameter must be a scalar or a two-element list/tuple/array.")


import numpy as np

import numpy as np

def make_transition_matrix(num_states, state_transition, trans_mat=None):
    if num_states < 2:
        raise ValueError("num_states must be at least 2.")

    if state_transition == "custom":
        if trans_mat is None:
            raise ValueError("trans_mat must be provided when state_transition='custom'.")
        trans_mat = np.asarray(trans_mat, dtype=float)

        if trans_mat.shape != (num_states, num_states):
            raise ValueError(
                f"trans_mat must have shape ({num_states}, {num_states}), got {trans_mat.shape}."
            )

        if not np.allclose(trans_mat.sum(axis=1), 1.0):
            raise ValueError("Each row of trans_mat must sum to 1.")

    elif state_transition == "uniform":
        trans_mat = np.ones((num_states, num_states), dtype=float) / num_states

    elif state_transition == "uniform_except_self":
        trans_mat = np.ones((num_states, num_states), dtype=float)
        np.fill_diagonal(trans_mat, 0.0)
        trans_mat /= (num_states - 1)

    elif state_transition == "return_to_baseline":
        trans_mat = np.zeros((num_states, num_states), dtype=float)
        trans_mat[0, 1:] = 1.0 / (num_states - 1)  # baseline -> others
        trans_mat[1:, 0] = 1.0                     # others -> baseline

    else:
        raise ValueError(
            "state_transition must be one of: "
            "'custom', 'uniform', 'uniform_except_self', 'return_to_baseline'."
        )

    return trans_mat

import numpy as np
from scipy.signal import sawtooth
from scipy.signal.windows import tukey


def _simulate_bursts(
    time_vec,
    fs,
    f,
    burst_cycles,
    noise_duration,
    burst_type,
    state_transition="uniform",
    transition_matrix=None,
    burst_amp_sigma=0.1,
    chi=0.15,
    use_tukey=True,
    tukey_alpha=0.25,
    power_law_scale=True,
    rng=None,
):
    """
    Simulate a signal consisting of alternating noise and burst segments.

    State 0 is reserved for noise, and states 1..K are burst states.

    Parameters
    ----------
    time_vec : array-like
        Time points for the simulation.
    fs : float
        Sampling frequency in Hz.
    f : scalar or array-like
        Burst frequency/frequencies in Hz.
        - If scalar, all burst states use the same frequency.
        - If array-like, its length defines the number of burst states.
    burst_cycles : scalar or length-2 sequence
        Number of cycles per burst.
        - If scalar, use a fixed number of cycles.
        - If length-2, draw an integer uniformly from [min, max].
    noise_duration : scalar or length-2 sequence
        Noise duration in seconds.
        - If scalar, use a fixed duration.
        - If length-2, draw uniformly from [min, max].
    burst_type : {"sine", "sawtooth"}
        Waveform used for burst segments.

    state_transition (str): Transition rule governing how the latent state
        sequence evolves over time.

        - "custom": use the user-specified `transition_matrix`.
        - "uniform": transition uniformly across all states, including the
          current state.
        - "uniform_except_self": transition uniformly across all states
          except the current state.
        - "return_to_baseline": treat state 0 as the baseline state; state 0
          transitions uniformly to nonzero states, and all nonzero states
          transition back to state 0.

    transition_matrix (array-like or None): Transition probability matrix
        used only when `state_transition="custom"`. It must have shape
        `(num_states, num_states)`, where each row specifies the
        probabilities of transitioning from one state to all possible next
        states.

    burst_amp_sigma : float
        Standard deviation of burst amplitude variation.
        Currently one amplitude is sampled per burst, not per cycle.
    chi : float
        Exponent for optional 1/f^chi amplitude scaling.
    use_tukey : bool
        Whether to apply a Tukey window to each burst.
    tukey_alpha : float
        Alpha parameter for the Tukey window.
    power_law_scale : bool
        Whether to scale burst amplitude by 1 / f^chi when multiple frequencies are used.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    signal : ndarray, shape (T,)
        Simulated signal.
    states : ndarray, shape (T,)
        Integer-valued state sequence. State 0 = noise, states 1..K = burst states.
    """
    if rng is None:
        rng = np.random.default_rng()

    freqs_array = np.atleast_1d(np.asarray(f, dtype=float))
    if np.any(freqs_array <= 0):
        raise ValueError("All frequencies in `f` must be positive.")

    use_scalar_freq = freqs_array.size == 1
    num_states = freqs_array.size + 1  # state 0 = noise
    init_p = np.ones(num_states, dtype=float) / num_states

    signal = np.zeros(len(time_vec), dtype=float)
    states = np.zeros(len(time_vec), dtype=int)

    trans_mat = make_transition_matrix(
        num_states=num_states,
        state_transition=state_transition,
        trans_mat=transition_matrix,
    )

    t = 0.0
    last_state = None

    while int(round(t * fs)) < len(time_vec):
        start_idx = int(round(t * fs))
        if start_idx >= len(time_vec):
            break

        # sample current state
        if last_state is None:
            current_state = rng.choice(num_states, p=init_p)
        else:
            current_state = rng.choice(num_states, p=trans_mat[last_state])

        last_state = current_state

        if current_state == 0:
            # ----- noise segment -----
            curr_noise_duration = _get_duration(noise_duration, rng)
            n_noise_samples = int(round(curr_noise_duration * fs))
            end_idx = min(start_idx + n_noise_samples, len(time_vec))

            states[start_idx:end_idx] = 0

            # advance using sample count to avoid drift
            t += (end_idx - start_idx) / fs

        else:
            # ----- burst segment -----
            freq_burst = freqs_array[0] if use_scalar_freq else freqs_array[current_state - 1]

            if isinstance(burst_cycles, (list, tuple, np.ndarray)) and len(burst_cycles) == 2:
                num_cycles = rng.integers(burst_cycles[0], burst_cycles[1] + 1)
            elif np.isscalar(burst_cycles):
                num_cycles = int(burst_cycles)
            else:
                raise ValueError("`burst_cycles` must be a scalar or a two-element sequence.")

            if num_cycles <= 0:
                raise ValueError("`burst_cycles` must be positive.")

            samples_per_cycle = int(round(fs / freq_burst))
            if samples_per_cycle <= 0:
                raise ValueError("Invalid samples_per_cycle. Check `fs` and `f`.")

            total_samples = samples_per_cycle * num_cycles
            aligned_time = np.arange(total_samples) / fs
            burst_amplitude = np.abs(rng.normal(loc=1.0, scale=burst_amp_sigma))

            if (not use_scalar_freq) and power_law_scale:
                burst_amplitude *= 1.0 / (freq_burst ** chi)

            if burst_type == "sine":
                burst_signal = burst_amplitude * np.sin(2 * np.pi * freq_burst * aligned_time)
            elif burst_type == "sawtooth":
                burst_signal = burst_amplitude * sawtooth(2 * np.pi * freq_burst * aligned_time, width=1)
            else:
                raise ValueError("Unsupported `burst_type`. Choose 'sine' or 'sawtooth'.")

            if use_tukey:
                burst_signal *= tukey(total_samples, alpha=tukey_alpha)

            end_idx = min(start_idx + total_samples, len(time_vec))
            burst_signal = burst_signal[: end_idx - start_idx]
            
            signal[start_idx:end_idx] = burst_signal
            states[start_idx:end_idx] = current_state

            # advance using exact written sample count
            t += (end_idx - start_idx) / fs

    return signal, states


def _generate_colored_noise(num_data, fs, beta=1, rng=None):
    """
    Generate colored noise with a 1/f^beta power spectrum.

    Parameters:
        beta (float): Exponent for the 1/f^beta distribution.
                      Use beta=0 for white noise, beta=1 for pink noise, beta=2 for brown noise.
        num_data (int): Number of samples in the noise signal.

    Returns:
        np.ndarray: The generated noise signal.

    Notes:
        - This function is based on neurodsp.sim.aperiodic.sim_powerlaw.[1]
        - The original reference is [2]

    References:
        [1] Cole, S., Donoghue, T., Gao, R., & Voytek, B. (2019). NeuroDSP: A package for
neural digital signal processing. Journal of Open Source Software, 4(36), 1272.
DOI: 10.21105/joss.01272. https://neurodsp-tools.github.io/neurodsp/_modules/neurodsp/sim/aperiodic.html#sim_powerlaw
        [2] Timmer, J., & Konig, M. (1995). On Generating Power Law Noise.
           Astronomy and Astrophysics, 300, 707–710.
    """

    if rng is None:
        rng = np.random.default_rng()

    # Generate white noise in the time domain.
    white_noise = rng.standard_normal(num_data)

    # Transform to frequency domain using the real FFT.
    spectrum = np.fft.rfft(white_noise)

    # Create frequency array; np.fft.rfftfreq returns frequencies for the rFFT.
    f = np.fft.rfftfreq(num_data, 1 / fs)

    spectrum[1:] /= f[1:] ** (beta / 2)  # |spectrum| ∝ 1/f^(β/2) (hence, power ∝ 1/f^β)
    spectrum[0] = 0

    # Transform back to the time domain.
    colored_noise = np.fft.irfft(spectrum, n=num_data).real
    #colored_noise = (colored_noise - colored_noise.mean()) / colored_noise.std()

    return np.array(colored_noise)


def _add_noise(bursts, states, noise, snr_db, use_filter = True, fs = None, highpass_f = 0.5):
    """
    Mix a clean burst signal with noise to achieve a target SNR (dB).

    Parameters
    ----------
    bursts : 1-D array
        Array that already contains bursts *and* zeros where there is noise.
    states : 1-D int array
        Parallel state vector (0 = noise gaps, >0 = burst indices).
    noise : 1-D array
        Noise signal (same length as bursts).
    snr_db : float
        Desired SNR, in decibels, defined as
            10·log10( signal_power / noise_power ).

    Returns
    -------
    noisy : 1-D array
        bursts + scaled_noise
    """

    if bursts.shape != noise.shape or bursts.shape != states.shape:
        raise ValueError("bursts, noise, and states must have identical shapes")
    if use_filter and (fs is None or highpass_f is None):
        raise ValueError("`fs` and `filter_frequency` must be provided when use_filter=True")


    bursts_copy = bursts.copy()
    noise_copy = noise.copy()  # this will be scaled based on SNR below
    all_states = np.unique(states)

    if use_filter:
        filtered_noise = filter_signal(noise_copy, fs, 'highpass', f_range= (highpass_f, None), remove_edges = False)

    else:
        filtered_noise = noise_copy

    noise_power = np.var(filtered_noise)
    for s in all_states:

        mask = states == s

        if s == 0:
            burst_power = np.var(bursts_copy[states != 0])

        else:
            burst_power = np.var(bursts_copy[mask])


        # desired noise power for that SNR
        snr_linear = 10 ** (snr_db / 10)
        desired_burst_power = snr_linear * noise_power

        # Scale bursts
        scaling_factor = np.sqrt(desired_burst_power / burst_power)
        bursts_copy[mask] *= scaling_factor

    return bursts_copy + noise_copy, bursts_copy




def simulate_bursty_signal(
        time_vec,
        fs,
        freq,
        burst_cycles_param,
        noise_duration_param,
        state_transition = 'uniform',
        transition_matrix = None,
        burst_type="sine",
        use_filter = True,
        highpass_f = 0.5,
        snr_db=0,
        burst_amp_sigma=0.1,
        beta=1,
        chi=0.15,
        use_tukey=True,
        tukey_alpha=0.25,
        rng=None):
    """
    Simulate a bursty signal with added noise.

    """

    if rng is None:
        rng = np.random.default_rng()

    # Generate bursts using the specified burst type and amplitude scale.
    bursts, states = _simulate_bursts(
        time_vec, fs, freq,
        burst_cycles_param, noise_duration_param,state_transition = state_transition, transition_matrix=transition_matrix,
        burst_type=burst_type, burst_amp_sigma=burst_amp_sigma, chi=chi,
        use_tukey=use_tukey, tukey_alpha=tukey_alpha, rng=rng
    )

    # Generate colored noise
    noise = _generate_colored_noise(len(time_vec), fs, beta, rng=rng)

    # scale noise based on the desired SNR
    signal, scaled_bursts = _add_noise(bursts, states, noise, snr_db, use_filter = use_filter, fs = fs, highpass_f = highpass_f)


    return {"signal": signal,
            "states": states,
            "bursts": scaled_bursts,
            "unscaled_bursts": bursts,
            "noise": noise,}


def simulate_empty_signal(time_vec):
    signal = np.zeros(len(time_vec), dtype=float)
    states = np.zeros(len(time_vec), dtype=int)
    bursts = np.zeros(len(time_vec), dtype=float)
    noise = np.zeros(len(time_vec), dtype=float)
    
    return {"signal": signal,
            "states": states,
            "bursts": bursts,
            "unscaled_bursts": bursts,
            "noise": noise,}


def simulate_noise(time_vec, fs, beta=1, rng=None):
    noise = _generate_colored_noise(len(time_vec), fs, beta, rng=rng)
    signal = np.copy(noise)
    states = np.zeros(len(time_vec), dtype=int)
    bursts = np.zeros(len(time_vec), dtype=float)

    return {"signal": signal,
            "states": states,
            "bursts": bursts,
            "noise": noise,}


def shift_right_overwrite(arr, s, n, x):
    """
    Shift n elements of list `arr` starting at index s to the right by x positions in-place.
    Overwrites destination positions. Vacated positions inside the affected window are filled with 0.
    If x <= 0 or n <= 0 does nothing. If x >= n, the n positions starting at s become 0.
    Indices are 0-based. Values that move past the end of arr are dropped.
    """
    length = len(arr)
    s = int(s)
    n = int(n)
    x = int(x)
    if n <= 0 or x <= 0 or s >= length:
        return

    # clamp start and n to valid window inside array
    if s < 0:
        # if start is negative, adjust n and start to operate inside array
        n += s  # reduce n by the negative offset
        s = 0
    n = max(0, min(n, length - s))

    if n == 0:
        return

    if x >= n:
        # whole window becomes zeros
        for i in range(s, s + n):
            arr[i] = 0
        return

    # move from right to left to avoid overwriting source values
    for i in range(s + n - 1, s - 1, -1):
        dest = i + x
        if dest < length:
            arr[dest] = arr[i]
        # else: moved out of bounds -> dropped

    # fill vacated positions within the window [s, s+x-1] with 0
    for i in range(s, min(s + x, length)):
        arr[i] = 0

def simulate_independent_signal(
        time_vec,
        fs,
        freq,
        burst_cycles_param,
        noise_duration_param,
        prev_states,
        state_transition = 'uniform',
        transition_matrix = None,
        burst_type="sine",
        use_filter = True,
        highpass_f = 0.5,
        snr_db=0,
        burst_amp_sigma=0.1,
        beta=1,
        chi=0.15,
        use_tukey=True,
        tukey_alpha=0.25,
        rng=None
        ):
    """
    Simulate a bursty signal with added noise.

    """

    if rng is None:
        rng = np.random.default_rng()

    bursts, states = _simulate_bursts(
        time_vec, fs, freq,
        burst_cycles_param, noise_duration_param,state_transition = state_transition, transition_matrix=transition_matrix,
        burst_type=burst_type, burst_amp_sigma=burst_amp_sigma, chi=chi,
        use_tukey=use_tukey, tukey_alpha=tukey_alpha, rng=rng
    )
    

    made_change = True
    overlap_start = 0
    overlap_end = 0
    overlap_found = False
    
    while made_change:
        made_change = False
        for p in prev_states:
            for index, (s1, s2) in enumerate(zip(p, states)):
                if (s1 == s2) and (s1 != 0):
                    if not overlap_found:
                        overlap_start = index
                        overlap_found = True
                else:
                    if overlap_found:
                        local_index = index
                        cur_s = 1
                        while cur_s != 0:
                            if local_index < len(states):
                                cur_s = states[local_index]
                            else:
                                break
                            local_index += 1
                        shift_right_overwrite(states, overlap_start, index-overlap_start, local_index-overlap_start)
                        shift_right_overwrite(bursts, overlap_start, index-overlap_start, local_index-overlap_start)
                        overlap_found = False
                        made_change = True
                        print("did a shift")
                    


    # Generate colored noise
    noise = _generate_colored_noise(len(time_vec), fs, beta, rng=rng)

    # scale noise based on the desired SNR
    signal, scaled_bursts = _add_noise(bursts, states, noise, snr_db, use_filter = use_filter, fs = fs, highpass_f = highpass_f)


    return {"signal": signal,
            "states": states,
            "bursts": scaled_bursts,
            "unscaled_bursts": bursts,
            "noise": noise,}


def phase_shift(signal_dict, degree, snr_db, fs, freq, use_filter=True, highpass_f=0.5):
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





def copy_signal_change_burst_freq(signal_dict, snr_db, fs, freq, rng, scale_amps=True, use_filter=True, highpass_f=0.5, chi=0.15, burst_amp_sigma=0.1, power_law_scale=True):
    states = np.copy(signal_dict["states"])
    bursts = np.copy(signal_dict["unscaled_bursts"])
    noise = np.copy(signal_dict["noise"])

    
    onset_start = 0
    onset_found = False
    for index, state in enumerate(states):
        if state != 0:
            if not onset_found:
                onset_start = index
                onset_found = True
        else:
            if onset_found:
                 
                

                use_scalar_freq = len(freq) == 1
                burst = np.copy(bursts[onset_start:index-1])
                aligned_time = np.arange(index-1-onset_start) / fs
                burst_amplitude = np.abs(rng.normal(loc=1.0, scale=burst_amp_sigma))
    


                if (not use_scalar_freq) and power_law_scale:
                    burst_amplitude *= 1.0 / (freq[states[index-1]-1] ** chi)
               
                burst = burst_amplitude * np.sin(2 * np.pi * freq[states[index-1]-1] * aligned_time)
                
                #adding amplitude variations

                cuts = np.sort(np.random.choice(np.arange(1, len(burst)), size=2, replace=False))
                amps = np.empty(len(burst), dtype=float)

                amps[:cuts[0]] = rng.uniform(0.7, 1.3)
                amps[cuts[0]:cuts[1]] = rng.uniform(0.7, 1.3)
                amps[cuts[1]:] = rng.uniform(0.7,1.3)

                if scale_amps: burst = burst * amps

                bursts[onset_start:index-1] = np.copy(burst)

                onset_found = False

    

    signal, scaled_bursts = _add_noise(bursts, states, noise, snr_db, use_filter = use_filter, fs = fs, highpass_f = highpass_f)


    return {"signal": signal,
            "states": states,
            "bursts": scaled_bursts,
            "unscaled_bursts": bursts,
            "noise": noise,}


def copy_signal_change_noise(signal_dict, snr_db, fs, rng, time_vec, use_filter=True, highpass_f=0.5, beta=1):
    states = np.copy(signal_dict["states"])
    bursts = np.copy(signal_dict["unscaled_bursts"])

    noise = _generate_colored_noise(len(time_vec), fs, beta, rng=rng)

    signal, scaled_bursts = _add_noise(bursts, states, noise, snr_db, use_filter = use_filter, fs = fs, highpass_f = highpass_f)
    
    return {"signal": signal,
            "states": states,
            "bursts": scaled_bursts,
            "unscaled_bursts": bursts,
            "noise": noise,}


def delay_bursts(signal_dict, snr_db, delay, fs, use_filter=True, highpass_f=0.5):
    
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



