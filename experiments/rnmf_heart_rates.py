import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tqdm.auto import tqdm

from utils import load_data_echonet


def calculate_energy(X, W, H, S, reg):
    return np.linalg.norm(X - W @ H - S, ord='fro') + reg*np.linalg.norm(S, ord=1)
    
def calcualte_low_rank_approximation(clip, reg=1e-1, eta=1e-6):
    
    # build data matrix
    X = []
    for frame in clip:
        X.append(frame.flatten())
    X = np.array(X)
    X = X.T
    
    # run iterative threasholding
    
    # hyperparameters
    k = 2 # rank of RNMF
    N = 1000
    
    # initialisation
    i = 0
    W = np.random.random(size=[X.shape[0], k])
    H = np.random.random(size=[k, X.shape[1]])
    S = np.zeros_like(X)
    energy = np.inf
    
    converged = False
    while not converged:

        energy_old = energy
        
        S = X - W @ H
        S = np.where(S > reg/2, reg - reg/2, 0)
        
        W_new = W * (np.maximum(X - S, 0) @ H.T)/(W @ H @ H.T + 1e-10)
        H_new = H * (W.T @ np.maximum(X - S, 0))/(W.T @ W @ H + 1e-10)
        
        W = W_new
        H = H_new
        
        W = W/np.linalg.norm(W, ord='fro')
        H = H/np.linalg.norm(H, ord='fro')
        
        i += 1
        energy = calculate_energy(X, W, H, S, reg)
        if i > N or np.abs(energy_old - energy) < eta:
            converged = True
    
    return H

def sine_model(x, frequency, amplitude, phase, offset, trend):
    return np.sin(2*np.pi*x*frequency + phase)*amplitude + offset + trend*x

def sine_model_jacobian(x, frequency, amplitude, phase, offset, trend):
    d_frequency = 2*np.pi*x*amplitude*np.cos(2*np.pi*x*frequency + phase)
    d_amplitude = np.sin(2*np.pi*x*frequency + phase)
    d_phase = amplitude*np.cos(2*np.pi*x*frequency + phase)
    d_offset = np.ones_like(x)
    d_trend = x
    return np.vstack([d_frequency, d_amplitude, d_phase, d_offset, d_trend]).T

def sine_fit(times, values):
    initial_amplitude = 0.5*(np.max(values) - np.min(values))
    initial_offset = np.mean(values)
    
    # try to fit sine for 4 different initial phases
    fits = []
    for initial_rate in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]:
        for initial_phase in [0.0, 0.5*np.pi, np.pi, 1.5*np.pi]:
            try:
                result = curve_fit(sine_model, times, values, p0=[initial_rate, initial_amplitude, initial_phase, initial_offset, 0.0], jac=sine_model_jacobian)
                fits.append(result[0])
            except:
                pass
    
    # no successful fit
    if len(fits) == 0:
        return None
    
    # normalise frequency to positive reals
    for i, fit in enumerate(fits):
        if fit[0] < 0.0:
            # change phase sign
            fits[i][2] *= -1
            # change amplitude sign
            fits[i][1] *= -1
    
    # normalise amplitude to positive reals
    for i, fit in enumerate(fits):
        if fit[1] < 0.0:
            # make amplitude positive
            fits[i][1] *= -1
            # adjust shift by pi
            fits[i][2] += np.pi
    
    # bring phases to range [0, 2pi]
    fits = np.array(fits)
    fits[:, 2] = np.mod(fits[:, 2], 2*np.pi)
    
    errors = [np.mean((sine_model(times[1:], *fit) - values[1:])**2) for fit in fits]
    return fits[np.argmin(errors)]


# Load EchoNet-Dynamic data
print('Load EchoNet-Dynamic data...')
echonet_info, files = load_data_echonet()
print('EchoNet-Dynamic data loaded.')

# For each subject: determine RNMF heart rate
print('RNMF heart rate detection...')
rnmf_rates = dict()
for id, filepath in tqdm(files.items()):

    print(id)

    data = np.load(filepath)
    times = data['times']
    frames = data['frames']

    H = calcualte_low_rank_approximation(frames)

    h0_params = sine_fit(times[1:], H[0, 1:])
    heart_rate0 = 60*h0_params[0]
    h1_params = sine_fit(times[1:], H[1, 1:])
    heart_rate1 = 60*h1_params[0]
    heart_rate = 0.5*(heart_rate0 + heart_rate1)
    
    if np.abs(heart_rate0 - heart_rate1)/min(heart_rate0, heart_rate1) < 0.1 and heart_rate >= 45 and heart_rate <= 180:
        rnmf_rates[id] = heart_rate

rnmf_df = pd.DataFrame.from_dict(rnmf_rates, orient='index', columns=['bpm'])
rnmf_df.to_csv('data/rnmf_heart_rates_echonet.csv')