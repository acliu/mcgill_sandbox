from pyuvdata import UVData
import hera_pspec as hp
import numpy as np
from scipy.stats import lognorm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import copy, os, itertools, inspect
from hera_pspec.data import DATA_PATH


dfile = os.path.join(DATA_PATH, 'eorsky_3.00hours_Nside128_sigma0.03_fwhm12.13_uv.uvh5')
uvd = UVData()
uvd.read(dfile)

print(uvd.get_antpairs())
print(uvd.Nbls,uvd.Nfreqs, uvd.Nspws, uvd.Ntimes)

cosmo = hp.conversions.Cosmo_Conversions()
print(cosmo)


# Creating the beamfile
beam_freqs = uvd.freq_array # in Hz
uvb = hp.PSpecBeamGauss(fwhm=0.21170844, beam_freqs=beam_freqs[0])


# Converting units
Jy_to_mK = uvb.Jy_to_mK(np.unique(uvd.freq_array), pol='pI')
uvd.data_array *= Jy_to_mK[None, None, :, None]


# slide the time axis of uvd by one integration
uvd1 = uvd.select(times=np.unique(uvd.time_array)[:-1:2], inplace=False)         #even
uvd2 = uvd.select(times=np.unique(uvd.time_array)[1::2], inplace=False)          #odd


# Create a new PSpecData object, and don't forget to feed the beam object
ds = hp.PSpecData(dsets=[uvd1, uvd2], wgts=[None, None], beam=uvb)


ds.rephase_to_dset(0)


# change units of UVData objects
ds.dsets[0].vis_units = 'mK'
ds.dsets[1].vis_units = 'mK'


# Specify which baselines to include
baselines = [(0, 11), (0, 12), (11, 12)]


# Define uvp
# Polarization pairs for this specific simulation data is 'pI'
# spw_range length = number of delays
uvp = ds.pspec(baselines, baselines, (0, 1), [('pI', 'pI')], spw_ranges=(0, 200), input_data_weight='identity',
               norm='I', taper='blackman-harris', verbose=True)   


# plot power spectrum of spectral window 1
fig, ax = plt.subplots(figsize=(12,8))

spw = 0
blp = ((0, 11), (0, 11))
key = (spw, blp, 'pI')
dlys = uvp.get_dlys(spw) * 1e9
power = np.abs(np.real(uvp.get_data(key)))

p1 = ax.plot(dlys, power.T)
ax.set_yscale('log')
ax.grid()
ax.set_xlabel("delay [ns]", fontsize=14)
ax.set_ylabel(r"$P(k)\ \rm [mK^2\ h^{-3}\ Mpc^3]$", fontsize=14)
ax.set_title("spw : {}, blpair : {}, pol : {}".format(*key), fontsize=14)


# select values recorded at different time
def select_time(ini_t, fin_t):
    p = []
    for i in range(int(fin_t-ini_t)+1):
        p.append(power[int(i+ini_t)])
    return np.transpose(p)

# plot the power sepctra at selected time
p = select_time(0, 3) # 0 is the first one; max 491; both included

fig, ax = plt.subplots(figsize=(12,8))
_ = ax.plot(dlys, p)
ax.set_yscale('log')
ax.grid()
ax.set_xlabel("delay [ns]", fontsize=14)
ax.set_ylabel(r"$P(k)\ \rm [mK^2\ h^{-3}\ Mpc^3]$", fontsize=14)
ax.set_title("spw : {}, blpair : {}, pol : {}".format(*key), fontsize=14)


# average the power spectrum
# form the baseline-pair group, which will be a single group 
# consisting of all baseline-pairs in the object
blp_group = [sorted(np.unique(uvp.blpair_array))]

# average spectra with inplace = False and assign to a new "uvp2" object
uvp2 = uvp.average_spectra(blpair_groups=blp_group, time_avg=True, inplace=False)

# plot power spectrum of spw 0
fig, ax = plt.subplots(figsize=(12,8))

spw = 0
blp = ((0, 11), (0, 11))
pol = 'pI'
key = (spw, blp, pol)
k_para = uvp.get_kparas(spw)
power = np.abs(np.real(uvp.get_data(key)))

avg_power = np.abs(np.real(uvp2.get_data(key)))

_ = ax.plot(k_para, power.T)
ax.plot(k_para, avg_power.T, color='k')
ax.set_yscale('log')
ax.grid()
ax.set_xlabel(r"$k_\parallel\ \rm h\ Mpc^{-1}$", fontsize=14)
ax.set_ylabel(r"$P(k)\ \rm [mK^2\ h^{-3}\ Mpc^3]$", fontsize=14)
ax.set_title("spw : {}, blpair : {}, pol : {}".format(*key), fontsize=14)


# histogram of P(k)
# return the nearest delay (in the dlys array) from the rough input delay
def get_delay(dly, key):
    dlys = uvp.get_dlys(key[0]) * 1e9
    dlys = np.asarray(dlys)
    idx = (np.abs(dlys - dly)).argmin()
    return dlys[idx]

# create a list of P(k) in a chosen delay interval or at a specific delay
def power_hist(i_dly, f_dly, key):
    power = np.real(uvp.get_data(key))
    ini_dly = get_delay(i_dly, key)
    fin_dly = get_delay(f_dly, key)
    dlys = uvp.get_dlys(key[0]) * 1e9
    i_index = int(np.where(dlys == ini_dly)[0])
    f_index = int(np.where(dlys == fin_dly)[0])
    y = []
    for i in range(int(f_index-i_index)+1):
        for j in range(len(power[i])):
            y.append(power[int(i+i_index)][j])
    return(y)

# plot a histogram of P(k)
fig, ax = plt.subplots(figsize=(12,8))

spw = 0
blp = ((0, 11), (0, 11))
pol = 'pI'
dly = [-1000, 2000]
key = (spw, blp, pol)
ini_dly = int(get_delay(dly[0], key))
fin_dly = int(get_delay(dly[1], key))
txt = (spw, blp, pol, ini_dly, fin_dly)

p = power_hist(dly[0], dly[1], key)

_ = ax.hist(p, bins='auto')
ax.grid()
ax.set_xlabel(r"$P(k)\ \rm [mK^2\ h^{-3}\ Mpc^3]$", fontsize=14)
ax.set_ylabel(r"Count", fontsize=14)
ax.set_title("spw : {}, blpair : {}, pol : {}, dlys : {}ns ~ {}ns".format(*txt), fontsize=14)


# CDF of the histogram
# plot cumulative distribution function
fig, ax = plt.subplots(figsize=(12,8))

spw = 0
blp = ((0, 11), (0, 11))
pol = 'pI'
dly = [-1000, 2000]
key = (spw, blp, pol)
ini_dly = int(get_delay(dly[0], key))
fin_dly = int(get_delay(dly[1], key))
txt = (spw, blp, pol, ini_dly, fin_dly)

p = power_hist(dly[0], dly[1], key)
cdf = np.array(range(len(p)))/float(len(p))

n, bins, patches = ax.hist(p, bins='auto', density=True, histtype='step', cumulative=True)
ax.plot(np.sort(p), cdf, 'k--')
# ax.hist(p, bins=bins, density=True, histtype='step', cumulative=-1)
ax.grid()
ax.set_xlabel(r"$P(k)\ \rm [mK^2\ h^{-3}\ Mpc^3]$", fontsize=14)
ax.set_ylabel(r"Probability", fontsize=14)
ax.set_title("spw : {}, blpair : {}, pol : {}, dlys : {}ns ~ {}ns".format(*txt), fontsize=14)


# fit the CDF to a lognormal CDF
def log_fct(x, mu, sigma):
    m = mu
    s = sigma
    return np.exp(-(np.log(x)-m)**2/(2*s**2))/(s*x*np.sqrt(2*np.pi))
popt, pcov = curve_fit(log_fct, np.sort(p), cdf)