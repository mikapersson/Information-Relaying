# Experimenting with the communications model
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

## Define probability of detection function
def probability_detect(eta, SNR):
    norm_cdf_arg = np.log(eta)/np.sqrt(SNR) - np.sqrt(SNR)/2
    cdf_value = norm.cdf(norm_cdf_arg)
    pd = 1 - cdf_value

    return pd

def cumulative_normal_arg(eta, SNR):
    norm_cdf_arg = np.log(eta)/np.sqrt(SNR) - np.sqrt(SNR)/2

    return norm_cdf_arg

def get_SNR(Pt, R, sigma2_n, phi_t=0, phi_r=0):
    SNR = Pt*np.cos(phi_t)*np.cos(phi_r) / (R**2 * sigma2_n)

    return SNR


""" SNR = 0.01
Pd = 0.01
print(np.exp((norm.ppf(1-Pd)+np.sqrt(SNR)/2) * np.sqrt(SNR))) """

""" eta = 1.268  # experiment with Pd = 0.01, R=10000, sigma2_n=0.01, Pt = 100
SNR = 0.01
print(1-norm.cdf(np.log(eta)/np.sqrt(SNR) - np.sqrt(SNR)/2)) """

# PLOTTING PD vs SNR
eta = np.array([1.2, 2, 5, 10])  # likelihood-ratio test threshold in [0,1]
sigma2_n = 1  # noise power
SNR = np.arange(0.01, 15, step=0.01)

Pds = []
for eta_i in eta:
    pd = np.array([probability_detect(eta_i, snr_i) for snr_i in SNR]) 
    Pds.append(pd)

    #arg = np.array([cumulative_normal_arg(eta[0], snr_i) for snr_i in SNR]) 

for i, eta_i in enumerate(eta):
    pd = Pds[i]
    plt.plot(SNR, pd, label=f'η = {eta_i}')  # Add a label for clarity
    #plt.plot(SNR, arg, label=f'η = {eta[0]}')  # Add a label for clarity

plt.xlabel('SNR (Signal-to-Noise Ratio)')  # x-axis label
plt.ylabel(r'$P_D$ [%]')  # y-axis label
plt.title('Probability of Detection vs SNR')  # Plot title
plt.axis([0, np.max(SNR), 0, 1])
plt.legend(loc="best")  # Show the legend
plt.grid()  # Add a grid for better visualization
plt.show()  # Display the plot

## PLOTTING PD vs R
""" eta = np.array([1.2, 2, 5])  # likelihood-ratio test threshold in [0,1]
Pt = 1000
sigma2_n = 0.0001  # noise power
Rs = np.arange(1, 30000, step=1)

Pds = []
for eta_i in eta:
    pd = np.array([probability_detect(eta_i, get_SNR(Pt, R, sigma2_n)) for R in Rs]) 
    Pds.append(pd)

    #arg = np.array([cumulative_normal_arg(eta[0], snr_i) for snr_i in SNR]) 

for i, eta_i in enumerate(eta):
    pd = Pds[i]
    plt.plot(Rs, pd, label=f'η = {eta_i}')  # Add a label for clarity
    #plt.plot(SNR, arg, label=f'η = {eta[0]}')  # Add a label for clarity

plt.xlabel('R [m]')  # x-axis label
plt.ylabel(r'$P_D$ [%]')  # y-axis label
plt.title('Probability of detection vs range')  # Plot title
plt.axis([0, np.max(Rs), 0, 1])
plt.legend(loc="best")  # Show the legend
plt.grid()  # Add a grid for better visualization
plt.show()  # Display the plot """
