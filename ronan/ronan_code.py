#!/usr/bin/env python
# coding: utf-8

# # Simple power spectrum estimation from an input dataset

# In[1]:

import matplotlib
matplotlib.use('Agg')

from pyuvdata import UVData
import hera_pspec as hp
import numpy as np
import matplotlib.pyplot as plt
import copy, os, itertools, inspect
from hera_pspec.data import DATA_PATH
import scipy


# ## Loading the input data: forming power spectra from adjacent time integrations
# 
# The input data are specified as a list of `UVData` objects, which are then packaged into a `PSpecData` class. This class is responsible for collecting the data and covariances together and performing the OQE power spectrum estimation.
# 
# At least two `UVData` objects must be specified, these could be different datasets, or just copies of a single dataset, given the use-case. In what follows, we will use only one data set and produce two copies of it, but will shift the second dataset by one time integration and interleave it relative to the first, such that we can form auto-baseline power spectra without noise-bias. 

# In[2]:


# select the data file to load
dfile = os.path.join(DATA_PATH, 'eorsky_3.00hours_Nside128_sigma0.03_fwhm12.13_uv.uvh5')
#dfile = os.path.join(DATA_PATH, 'zen.all.xx.LST.1.06964.uvA')

# Load into UVData objects
uvd = UVData()
uvd.read(dfile)


# In[37]:


# Check which baselines are available
print(uvd.Nfreqs)
print(uvd.get_antpairpols())


# ## Define a cosmology

# In[4]:


# Instantiate a Cosmo Conversions object
# we will need this cosmology to put the power spectra into cosmological units
cosmo = hp.conversions.Cosmo_Conversions()
print(cosmo)


# Instantiate a beam object, and attach the cosmo conversions object onto it.

# In[5]:


# List of beamfile to load. This is a healpix map.
beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
beam_freqs = np.linspace(0, 384e6, 384)
# intantiate beam and pass cosmology, if not fed, a default Planck cosmology will be assumed
#uvb = hp.pspecbeam.PSpecBeamUV(beamfile, cosmo=cosmo)
uvb = hp.PSpecBeamGauss(fwhm=0.1, beam_freqs=beam_freqs)


# Next convert from Jy units to mK. This involves calculating the effective beam area (see HERA Memo #27 and #43), which can be done with the beam object we instantiated earlier.

# In[6]:


# find conversion factor from Jy to mK
Jy_to_mK = uvb.Jy_to_mK(np.unique(uvd.freq_array), pol='XX')

# reshape to appropriately match a UVData.data_array object and multiply in!
uvd.data_array *= Jy_to_mK[None, None, :, None]


# Configure data and instantiate a `PSpecData` object, while also feeding in the beam object.

# In[7]:


# slide the time axis of uvd by one integration
uvd1 = uvd.select(times=np.unique(uvd.time_array)[:-1:2], inplace=False)
uvd2 = uvd.select(times=np.unique(uvd.time_array)[1::2], inplace=False)

# Create a new PSpecData object, and don't forget to feed the beam object
ds = hp.PSpecData(dsets=[uvd1, uvd2], wgts=[None, None], beam=uvb)


# ## Phase second `dset` to first `dset`

# In[8]:


# Because the LST integrations are offset by more than ~15 seconds we will get a warning
# but this is okay b/c it is still **significantly** less than the beam-crossing time and we are using short
# baselines...

# here we phase all datasets in dsets to the zeroth dataset
ds.rephase_to_dset(0)


# In[9]:


# change units of UVData objects
ds.dsets[0].vis_units = 'mK'
ds.dsets[1].vis_units = 'mK'


# ## Estimating the power spectrum for a handful of baseline pairs (auto-baseline pspec)

# In[38]:


# Specify which baselines to include
baselines = [(0, 11), (0, 12), (11, 12)]
baselines1, baselines2, blpairs = hp.utils.construct_blpairs(baselines, exclude_auto_bls=True,exclude_permutations=True)


# In[14]:


# Here we either have the choice between forming pairs of the same baselines or
# to form pairs of different baselines measuring with same spectral window (redundant baselines).

#uvp = ds.pspec(baselines, baselines, (0, 1), [('pI', 'pI')], spw_ranges=[(0, 84)], input_data_weight='identity',norm='I', taper='blackman-harris', verbose=True)

uvp = ds.pspec(baselines1, baselines2, (0, 1), [('pI', 'pI')], spw_ranges=[(0, 84)], input_data_weight='identity',norm='I', taper='blackman-harris', verbose=True)


# In[39]:


# This is how we can get the delay spectra data. We can also verify the dimensions of the data set.

key = (0, ((0,11),(0,12)), ('pI', 'pI'))

# output should be shape (Ntimes, Ndlys)
print(uvp.get_data(key).shape)

# we can also access data by feeding a dictionary
key = {'polpair':('pI','pI'), 'spw': 0, 'blpair':((0, 11), (0, 12))}
print(uvp.get_data(key).shape)


# In[40]:


# get power spectrum units
print("pspec units: ", uvp.units)
# get weighting
print("pspec weighting: ", uvp.weighting)


# In[17]:


print(uvp.cosmo)


# ## Plotting

# In[41]:


# Here, we can plot power spectrums based on a spectra frequency range, a baseline pair and the polarization of our telescope.
fig, ax = plt.subplots(2,figsize=(12,11))

spw = 0
blp =((0, 11), (0,12))
key = (spw, blp, 'pI')
dlys = uvp.get_dlys(spw) * 1e9
power = np.abs(np.real(uvp.get_data(key)))

# We can also choose to plot the power spectrum over a range of different time values.
p1 = ax[0].plot(dlys, power.T[:,100:110])
ax[0].set_yscale('log')
ax[0].grid()
ax[0].set_xlabel("delay [ns]", fontsize=14)
ax[0].set_ylabel(r"$P(k)\ \rm [mK^2\ h^{-3}\ Mpc^3]$", fontsize=14)
ax[0].set_title("spw : {}, blpair : {}, pol : {}".format(*key), fontsize=14)

# If we wish to see the total power received from different delays, we can sum up the powers over all time.
p1 = ax[1].plot(dlys, power.T.sum(axis=1))
ax[1].set_yscale('log')
ax[1].grid()
ax[1].set_xlabel("delay [ns]", fontsize=14)
ax[1].set_ylabel("Total " + r"$P(k)\ \rm [mK^2\ h^{-3}\ Mpc^3]$", fontsize=14)
ax[1].set_title("spw : {}, blpair : {}, pol : {}".format(*key), fontsize=14)

plt.savefig("ps_total.png")
# In[42]:


# Here, we wish to calculate the difference in power between all possible combinations of baseline pairs over all times. 
# This will allow us to observe any differing power levels between two baseline configurations.

from itertools import combinations
i=0
for comb in combinations(uvp.get_blpairs(), 2):
    i=i+1
fig, ax_resi = plt.subplots(i,figsize=(12,14))
i=0
for comb in combinations(uvp.get_blpairs(), 2):
    
    power_A = np.abs(np.real(uvp.get_data(key=(spw, comb[0], 'pI'))))
    power_B = np.abs(np.real(uvp.get_data(key=(spw, comb[1], 'pI'))))
    power_residual_pairs = power_A-power_B
    
    ax_resi[i].plot(dlys, power_residual_pairs.T[:,400:410])
    ax_resi[i].set_title("Residuals between " + str(comb[0]) + " and " + str(comb[1]) + " powers", fontsize=14)
    ax_resi[i].grid()
    ax_resi[i].set_ylabel(r"$P(k)\ \rm [mK^2\ h^{-3}\ Mpc^3]$", fontsize=14)
    i=i+1

plt.savefig("resibaselines.png")
# In[43]:


from scipy.stats import norm

# We plot a histogram of power values over different times at a specific delay to give us insight on possible noise at that delay.

chosen_delay = -400

# The delay chosen might not exist in the data. In that case, we warn and choose the nearest delay value.

# This minimizes the separation from the closest value.
closest_delay = min(dlys, key=lambda x:abs(x-chosen_delay))

if chosen_delay != closest_delay:
	# Warn that the closest value to that will be chosen.
	print("There are no data points at " + str(chosen_delay) + ". The closest value to the desired delay is " + str(closest_delay))

# We find the index in the delay array, which will correspond to the same index where the power values are at.
delay_index = np.where(dlys==closest_delay)

data = power.T[delay_index].flatten()

fig_hist1, ax_hist1 = plt.subplots(2,figsize=(12,8))

p2 = ax_hist1[0].hist(data,bins=30,density=True,linewidth=2,edgecolor='k')

# We fit the data with a normal distribution and find the best fit parameters for the mean and standard deviation.
mu, std = norm.fit(data)
x = np.linspace(min(data), max(data), 30)
p = norm.pdf(x, mu, std)
ax_hist1[0].plot(x, p, 'k', linewidth=2)
ax_hist1[0].set_ylabel("Probability",fontsize=14)
ax_hist1[0].set_xlabel(r"$P(k)\ \rm [mK^2\ h^{-3}\ Mpc^3]$",fontsize=14)

# We plot the residuals of the histogram and the fit.
ax_hist1[1].plot(x,p2[0]-p)
ax_hist1[1].axhline(y=0, color='k', linestyle='-',linewidth=3)
ax_hist1[1].set_ylabel("Residuals",fontsize=14)
ax_hist1[1].set_xlabel(r"$P(k)\ \rm [mK^2\ h^{-3}\ Mpc^3]$",fontsize=14)

plt.savefig("histsingle.png")
# In[22]:


# We now want to plot a histogram of power values over a range of delay times.

from scipy.stats import skewnorm

# We choose the range of delays.
chosen_min_delay = -5000
chosen_max_delay = -4000
# And get the closest values if the chosen delays don't exist.
closest_min_delay = min(dlys, key=lambda x:abs(x-chosen_min_delay))
closest_max_delay = min(dlys, key=lambda x:abs(x-chosen_max_delay))

if chosen_min_delay != closest_min_delay:
	# Warn that the closest value to the minimum delay will be chosen.
	print("There are no data points at " + str(chosen_min_delay) + ". The closest value to the desired delay is " + str(closest_min_delay))

if chosen_max_delay != closest_max_delay:
	# Warn that the closest value to the maximum delay will be chosen.
	print("There are no data points at " + str(chosen_max_delay) + ". The closest value to the desired delay is " + str(closest_max_delay))

min_delay_index = np.where(dlys==closest_min_delay)
max_delay_index = np.where(dlys==closest_max_delay)

data=power.T[min_delay_index[0][0]:max_delay_index[0][0]].flatten()

fig_hist2, ax_hist2 = plt.subplots(2,figsize=(12,8))

x = np.linspace(min(data),max(data), 30)
# We now fit a skewed probability distribution.
ax_hist2[0].plot(x, skewnorm.pdf(x, *skewnorm.fit(data)),lw=3,color='k',label='skewnorm pdf')

p4 = ax_hist2[0].hist(data,bins=30,density=True,linewidth=2,edgecolor='k')
ax_hist2[0].set_ylabel("Probability",fontsize=14)
ax_hist2[0].set_xlabel(r"$P(k)\ \rm [mK^2\ h^{-3}\ Mpc^3]$",fontsize=14)

# We plot the residuals between our histogram and our fit.
ax_hist2[1].plot(x,p4[0]-skewnorm.pdf(x, *skewnorm.fit(data)))
ax_hist2[1].axhline(y=0, color='k', linestyle='-',linewidth=3)
ax_hist2[1].set_ylabel("Residuals",fontsize=14)
ax_hist2[1].set_xlabel(r"$P(k)\ \rm [mK^2\ h^{-3}\ Mpc^3]$",fontsize=14)

plt.savefig("histrange.png")
# In[35]:


# We now want to plot the cumulative distribution function based on our previous histograms.

fig_cdf, ax_cdf_delay_range = plt.subplots(figsize=(12,4))

data = np.sort(power.T[min_delay_index[0][0]:max_delay_index[0][0]].flatten())

cdf = np.cumsum(data) # calculate the cdf

p3 = ax_cdf_delay_range.plot(data,cdf)

ax_cdf_delay_range.set_xlabel(r"$P(k)\ \rm [mK^2\ h^{-3}\ Mpc^3]$", fontsize=14)
ax_cdf_delay_range.set_ylabel("Cumulative probability", fontsize=14)

plt.savefig("cdfrange.png")
# In[34]:


fig_cdf, ax_cdf_single_delay = plt.subplots(figsize=(12,4))

data = np.sort(power.T[delay_index].flatten())

cdf =  np.cumsum(data)# calculate the cdf

p3 = ax_cdf_single_delay.plot(data,cdf)

ax_cdf_single_delay.set_xlabel(r"$P(k)\ \rm [mK^2\ h^{-3}\ Mpc^3]$", fontsize=14)
ax_cdf_single_delay.set_ylabel("Cumulative probability", fontsize=14)

plt.savefig("cdfsingle.png")
# In[ ]:





