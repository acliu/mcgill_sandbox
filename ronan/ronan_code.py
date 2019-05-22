# # Simple power spectrum estimation from an input dataset

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

# select the data file to load
dfile = os.path.join(DATA_PATH, 'eorsky_3.00hours_Nside128_sigma0.03_fwhm12.13_uv.uvh5')
#dfile = os.path.join(DATA_PATH, 'zen.all.xx.LST.1.06964.uvA')

# Load into UVData objects
uvd = UVData()
uvd.read(dfile)

# Check which baselines are available
print(uvd.Nfreqs)
pol = uvd.get_antpairpols()[0][2]


# Instantiate a Cosmo Conversions object
# we will need this cosmology to put the power spectra into cosmological units
cosmo = hp.conversions.Cosmo_Conversions()
print(cosmo)


# Instantiate a beam object, and attach the cosmo conversions object onto it.
# List of beamfile to load. This is a healpix map.

beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
beam_freqs = np.linspace(0, 384e6, 384)

# intantiate beam and pass cosmology, if not fed, a default Planck cosmology will be assumed
#uvb = hp.pspecbeam.PSpecBeamUV(beamfile, cosmo=cosmo)

uvb = hp.PSpecBeamGauss(fwhm=0.1, beam_freqs=beam_freqs)


# Next convert from Jy units to mK. This involves calculating the effective beam area (see HERA Memo #27 and #43), which can be done with the beam object we instantiated earlier.

# find conversion factor from Jy to mK

Jy_to_mK = uvb.Jy_to_mK(np.unique(uvd.freq_array), pol=pol)

# reshape to appropriately match a UVData.data_array object and multiply in!

uvd.data_array *= Jy_to_mK[None, None, :, None]

# Configure data and instantiate a `PSpecData` object, while also feeding in the beam object.

# slide the time axis of uvd by one integration

uvd1 = uvd.select(times=np.unique(uvd.time_array)[:-1:2], inplace=False)
uvd2 = uvd.select(times=np.unique(uvd.time_array)[1::2], inplace=False)

# Create a new PSpecData object, and don't forget to feed the beam object
ds = hp.PSpecData(dsets=[uvd1, uvd2], wgts=[None, None], beam=uvb)

# Because the LST integrations are offset by more than ~15 seconds we will get a warning
# but this is okay b/c it is still **significantly** less than the beam-crossing time and we are using short
# baselines...

# here we phase all datasets in dsets to the zeroth dataset

ds.rephase_to_dset(0)

# change units of UVData objects

ds.dsets[0].vis_units = 'mK'
ds.dsets[1].vis_units = 'mK'

# Specify which baselines to include

baselines = [(0, 11), (0, 12), (11, 12)]
baselines1, baselines2, blpairs = hp.utils.construct_blpairs(baselines, exclude_auto_bls=True,exclude_permutations=True)

#We ask if the user wants to plot power spectrums using PS estimation code 
# or averages of visibilities as a simpler estimator of the power spectrum.
print("Choose between generating power spectrums with PS estimation code (0) or to work with mean squared averages of visibility data (1) :")
choice = input()

if choice == 0:
	
	# Here we either have the choice between forming pairs of the same baselines or
	# to form pairs of different baselines measuring with same spectral window (redundant baselines).

	#uvp = ds.pspec(baselines, baselines, (0, 1), [('pI', 'pI')], spw_ranges=[(0, 84)], input_data_weight='identity',norm='I', taper='blackman-harris', verbose=True)

	uvp = ds.pspec(baselines1, baselines2, (0, 1), [(pol, pol)], spw_ranges=[(0, 84)], input_data_weight='identity',norm='I', taper='blackman-harris', verbose=True)

	# This is how we can get the delay spectra data. We can also verify the dimensions of the data set.

	key = (0, ((0,11),(0,12)), (pol, pol))

	# output should be shape (Ntimes, Ndlys)

	print(uvp.get_data(key).shape)

	# we can also access data by feeding a dictionary

	key = {'polpair':(pol,pol), 'spw': 0, 'blpair':((0, 11), (0, 12))}
	print(uvp.get_data(key).shape)

	# get power spectrum units
	print("pspec units: ", uvp.units)
	# get weighting
	print("pspec weighting: ", uvp.weighting)

	print(uvp.cosmo)

	# Here, we can plot power spectrums based on a spectra frequency range, a baseline pair and the polarization of our telescope.
	# Unfortunately for the moment, noise generation and average power do not seem to work well with this new code.
	# So they were commented out.
	blpairs = uvp.get_blpairs()
	blp_group = [sorted(np.unique(uvp.blpair_array))]

	Tsys = 300
	color=['r','b','g']
	# We can also choose to plot the power spectrum over a range of different time values.

	# We can choose which result the user wants to see. Option 0 corresponds to the residuals between all possible baseline pairs,
	# option 1 corresponds to simply displaying the power spectrum of each baseline pairs in separate plots.
	# Option 2 corresponds to displaying power spectrum along side the average power and standard deviation of the power over time.

	print("Please select one of the following options : \n")
	print("Input 0 for power residuals between all baseline pairs. 1 to simply display all possible power spectrums.")
	print("2 to display power spectrum alongside a plot of the average power and standard deviation over time.")
	option = input()

	from itertools import combinations

	if option == 0 :
		print("Select spectral window index from " + str(len(uvp.spw_array)) + " choices : ")
		spw = input()
		dlys = uvp.get_dlys(spw) * 1e9
		
		# Here, we wish to calculate the difference in power between all possible combinations of baseline pairs over all times. 
		# This will allow us to observe any differing power levels between two baseline configurations.
		i=0
		for comb in combinations(uvp.get_blpairs(), 2):
			i=i+1
		fig, ax_resi = plt.subplots(i,figsize=(12,14))
		i=0
		for comb in combinations(uvp.get_blpairs(), 2):

			power_A = np.abs(np.real(uvp.get_data(key=(spw, comb[0], pol))))
			power_B = np.abs(np.real(uvp.get_data(key=(spw, comb[1], pol))))
			power_residual_pairs = power_A-power_B

			ax_resi[i].plot(dlys, power_residual_pairs.T[:,400:410])
			ax_resi[i].set_title("Residuals between " + str(comb[0]) + " and " + str(comb[1]) + " powers", fontsize=14)
			ax_resi[i].grid()
			ax_resi[i].set_ylabel(r"$P(k)\ \rm [mK^2\ h^{-3}\ Mpc^3]$", fontsize=14)
			i=i+1

	if option == 1:
		print("Select spectral window index from " + str(len(uvp.spw_array)) + " choices : ")
		spw = input()
		fig, ax = plt.subplots(len(blpairs),1,figsize=(12,14))
		for i in range(0,len(blpairs)):
			dlys = uvp.get_dlys(spw) * 1e9
			key = (spw, blpairs[i], pol)
			power = np.abs(np.real(uvp.get_data(key)))

			ax[i].plot(dlys, power.T[:,0:10])
			ax[i].set_yscale('log')
			ax[i].grid()
			ax[i].set_xlabel("delay [ns]", fontsize=14)
			ax[i].set_ylabel(r"$P(k)\ \rm [mK^2\ h^{-3}\ Mpc^3]$", fontsize=14)
			ax[i].set_title("spw : {}, blpair : {}, pol : {}".format(*key), fontsize=14)

		plt.tight_layout()

	if option == 2:
	    
	    fig, ax = plt.subplots(2*len(blpairs),1,figsize=(12,20))
	    for i in range(0,len(blpairs)):
	        k = 2*i
	        ax_std = ax[k+1].twinx()
	        for j in range(0,len(uvp.get_spw_ranges())):
	            dlys = uvp.get_dlys(j) * 1e9
	            key = (j, blpairs[i], pol)
	            power = np.abs(np.real(uvp.get_data(key)))
	            time_array = np.linspace(0,power.T.shape[1],power.T.shape[1])
	            #P_N = uvp.generate_noise_spectra(j, pol, Tsys)
	            #P_N = P_N[uvp.antnums_to_blpair(blpairs[i])]
	            #uvp2 = uvp.average_spectra(blpair_groups=blp_group, time_avg=True, inplace=False)
	            #avg_power = np.abs(np.real(uvp2.get_data(key)))

	            std_mean_all_time = np.mean(np.std(power.T[:,190:200],axis=0))/np.sqrt(10)
	            std_mean = np.std(np.mean(power.T[:,190:200],axis=1))

	            ax[k].plot(dlys, power.T[:,0:10])
	            #ax[i].plot(dlys, avg_power.T, color='k')
	            #ax[k].set_prop_cycle(None)
	            #ax[k].plot(dlys, P_N.T,color='k', ls='--', lw=3)

	            # If we wish to see the total power received from different delays, we can sum up the powers over all time.
	            ax[k+1].plot(time_array, np.mean(power.T,axis=0),ls='--',lw=3,label="Average power")

	            # We could also analyze the evolution of the standard deviation of the power at one time to look for patterns.

	            ax_std.plot(time_array, np.std(power.T,axis=0),lw=3,label="STD")
	        ax[k].set_yscale('log')
	        ax[k].grid()
	        ax[k].set_xlabel("delay [ns]", fontsize=14)
	        ax[k].set_ylabel(r"$P(k)\ \rm [mK^2\ h^{-3}\ Mpc^3]$", fontsize=14)
	        ax[k].set_title("spw : {}, blpair : {}, pol : {}".format(*key), fontsize=14)

	        ax[k+1].grid()
	        ax[k+1].set_xlabel("Time", fontsize=14)
	        ax[k+1].set_ylabel("Average " + r"$P(k)$", fontsize=14)
	        ax[k+1].set_title("spw : {}, blpair : {}, pol : {}".format(*key), fontsize=14)
	        ax[k+1].legend()
	        plt.tight_layout()
	        ax_std.set_ylabel("Standard deviation", fontsize=14)
	        ax_std.legend()
	        print("spw : {}, blpair : {}, pol : {}".format(*key) + " : Gaussian std (1/np.sqrt(N)) = " 
	              + str(std_mean_all_time) + ", Average spectra std = " + str(std_mean) + ", Ratio = " + str(std_mean_all_time/std_mean))


	from scipy.stats import norm

	# We plot a histogram of power values over different times at a specific delay to give us insight on possible noise at that delay.
	print("Enter which type of histogram to be displayed :")
	print("0 is for single delay histogram. 1 is for a wide range of delay histogram : ")
	option_hist = input()
	if option_hist == 0 :
		print("Enter the delay you want to display a histogram of the power values and the interval of time : ")
		print("Choose delay : ")
		chosen_delay = input()
		print("Choose minimum time : ")
		time_min = input()
		print("Choose maximum time : ")
		time_max= input()
		# The delay chosen might not exist in the data. In that case, we warn and choose the nearest delay value.

		# This minimizes the separation from the closest value.
		closest_delay = min(dlys, key=lambda x:abs(x-chosen_delay))

		if chosen_delay != closest_delay:
			# Warn that the closest value to that will be chosen.
			print("There are no data points at " + str(chosen_delay) + ". The closest value to the desired delay is " + str(closest_delay))

		print("Select spectral window index from " + str(len(uvp.spw_array)) + " choices : ")
		spw = input()
		print("Select baseline pair from " + str(uvp.get_blpairs()) + " by typing the index : ")
		i=input()
		key = (spw, blpairs[i], pol)
		power = np.abs(np.real(uvp.get_data(key)))
		# We find the index in the delay array, which will correspond to the same index where the power values are at.
		delay_index = np.where(dlys==closest_delay)

		data = power.T[delay_index][:,time_min:time_max].flatten()

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

		fig_cdf, ax_cdf_single_delay = plt.subplots(figsize=(12,4))

		cdf =  np.cumsum(p2[0])# calculate the cdf

		p3 = ax_cdf_single_delay.plot(p2[1][1:],cdf)

		ax_cdf_single_delay.set_xlabel(r"$P(k)\ \rm [mK^2\ h^{-3}\ Mpc^3]$", fontsize=14)
		ax_cdf_single_delay.set_ylabel("Cumulative probability", fontsize=14)


	# We now want to plot a histogram of power values over a range of delay times if
	# the user chose to do so.

	if option_hist == 1 :
		from scipy.stats import skewnorm
		print("Enter the range of delay you want to include in the histogram and the interval of time : ")
		# We choose the range of delays.
		print("Choose minimum delay : ")
		chosen_min_delay = input()
		print("Choose maximum delay : ")
		chosen_max_delay = input()
		print("Choose minimum time : ")
		time_min = input()
		print("Choose maximum time : ")
		time_max= input()

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

		print("Select spectral window index from " + str(len(uvp.spw_array)) + " choices : ")
		spw = input()
		print("Select baseline pair from " + str(uvp.get_blpairs()) + " by typing the index : ")
		i=input()
		key = (spw, blpairs[i], pol)
		power = np.abs(np.real(uvp.get_data(key)))

		data=power.T[min_delay_index[0][0]:max_delay_index[0][0]][:,time_min:time_max].flatten()

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

		# We now want to plot the cumulative distribution function based on our previous histograms.

		fig_cdf, ax_cdf_delay_range = plt.subplots(figsize=(12,4))

		cdf = np.cumsum(p4[0]) # calculate the cdf

		p3 = ax_cdf_delay_range.plot(p4[1][1:],cdf)

		ax_cdf_delay_range.set_xlabel(r"$P(k)\ \rm [mK^2\ h^{-3}\ Mpc^3]$", fontsize=14)
		ax_cdf_delay_range.set_ylabel("Cumulative probability", fontsize=14)

if choice == 1:
	import matplotlib.gridspec as gridspec
	data_type = {}
	print("This is for generating histograms and cdf of visibility functions at different frequency ranges.\n")
	print("How many ranges of frequency to split equally?")
	print("Total frequency is " + str(uvd.Nfreqs) + " :")
	num_intervals = input()
	intervals = np.linspace(0,uvd.Nfreqs,num_intervals)

	spw_ranges=[]
	for i in range(0,num_intervals-1):
	    spw_ranges.append((int(intervals[i]),int(intervals[i+1])))

	spw_ranges.append((0,int(max(intervals))))

	print("All frequency ranges : " + str(spw_ranges))

	fig = plt.figure(constrained_layout=False)

	functions = ["<|v1|^2>","<v1-v2>","<|v1*v2*|>","<|v1|^2>-<|s1|^2>"]
	#We will store the histogram values to plot the cdf
	cdf=[]
	c=0
	r=0
	num_rows = int(2*len(spw_ranges)/3)+2

	gs = gridspec.GridSpec(num_rows,3)
	gs.update(wspace=0.2, hspace=0.13)
	print("Select the index of the function to plot histograms from the choices below :")
	print(functions)
	output = input()
	for k in range(0,2):
	    
	    for j in range(0,len(spw_ranges)):

	        min_freq = spw_ranges[j][0]
	        max_freq = spw_ranges[j][1]
	        even_time = uvd.data_array[:,0,min_freq:max_freq,0].T[:,0:][:,::2]
	        odd_time = uvd.data_array[:,0,min_freq:max_freq,0].T[:,1:][:,::2]

	        if functions[output] == "<|v1|^2>":
	            data_type["<|v1|^2>"] = np.mean((np.absolute(uvd.data_array[:,0,min_freq:max_freq,0].T))**2,axis=1)

	        # This is subtracting all even and odd times with each other
	        if functions[output] == "<v1-v2>":
	            data_type["<v1-v2>"] = np.real(np.mean(even_time-odd_time,axis=1))
	        if functions[output] == "<|v1*v2*|>":
	            data_type["<|v1*v2*|>"] = np.mean(np.absolute(np.multiply(even_time,np.conjugate(odd_time))),axis=1)
	        if functions[output] == "<|v1|^2>-<|s1|^2>":
	            data_type["<|v1*v2*|>"] = np.mean(np.absolute(np.multiply(even_time,np.conjugate(odd_time))),axis=1)
	            data_type["<|v1|^2>-<|s1|^2>"] = np.mean((np.absolute(uvd.data_array[:,0,min_freq:max_freq,0].T))**2,axis=1)-data_type["<|v1*v2*|>"]

	        data = data_type.get(functions[output],"").flatten()

	        if c==3:
	            c=0
	            r=r+1

	        gss01 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[j+len(spw_ranges)*k],hspace=0.0)

	        ax0 = fig.add_subplot(gss01[0])
	        ax1 = fig.add_subplot(gss01[1], sharex=ax0)
	        if k == 0:
	            if j == len(spw_ranges)-1:
	                col='g'
	            else:
	                col='#1f77b4'
	            lab=functions[output] + str(spw_ranges[j])
	        else:
	            lab=functions[output] + str(spw_ranges[j]) + "log"
	            ax0.set_xscale('log')
	            ax1.set_xscale('log')
	            if j == len(spw_ranges)-1:
	                col='g'
	            else:
	                col='r'
	        p=ax0.hist(data,alpha=0.5,bins=int(30*5/len(spw_ranges)),density=False,linewidth=3,edgecolor='k',label=lab,color=col)
	        
	        cdf_sum = np.cumsum(p[0].flatten()) # calculate the cdf
	        ax1.plot(p[1][1:],cdf_sum,alpha=1,label=lab,color=col)
	        ax1.legend()   
	        c=c+1

	fig.set_size_inches(w=15,h=3*num_rows+1)

	print("Now the plots for different visibility functions will be generated :")

	fig, ax = plt.subplots(figsize=(12,8))

	ax.grid()
	ax.set_xlabel("Frequency", fontsize=14)
	ax.set_ylabel("Visibility", fontsize=14)

	data_type = {}
	even_time = uvd.data_array[:,0,:,0].T[:,0:][:,::2]
	odd_time = uvd.data_array[:,0,:,0].T[:,1:][:,::2]
	print("Which functions of visibility to display? Select by index. Type '10' to finish.")
	print(functions)
	output=input()
	while(output != 10):
	    if output != 10:
	        if functions[output] == "<|v1|^2>":
	                data_type["<|v1|^2>"] = np.mean((np.absolute(uvd.data_array[:,0,min_freq:max_freq,0].T))**2,axis=1)

	        # This is subtracting all even and odd times with each other
	        if functions[output] == "<v1-v2>":
	            data_type["<v1-v2>"] = np.real(np.mean(even_time-odd_time,axis=1))
	        if functions[output] == "<|v1*v2*|>":
	            data_type["<|v1*v2*|>"] = np.mean(np.absolute(np.multiply(even_time,np.conjugate(odd_time))),axis=1)
	        if functions[output] == "<|v1|^2>-<|s1|^2>":
	            data_type_v1v2 = np.mean(np.absolute(np.multiply(even_time,np.conjugate(odd_time))),axis=1)
	            data_type["<|v1|^2>-<|s1|^2>"] = np.mean((np.absolute(uvd.data_array[:,0,min_freq:max_freq,0].T))**2,axis=1)-data_type_v1v2
	    print("Which functions of visibility to display? Select by index. Type '10' to finish.")
	    print(functions)
	    output=input()

	print("Real part average v: " + str(np.mean(np.mean((uvd.data_array[:,0,:,0].T).real,axis=1))))
	print("Imaginary part average v: " + str(np.mean(np.mean(uvd.data_array[:,0,:,0].T.imag,axis=1))))
	c=['r','g','b','k']

	keys = data_type.keys()
	print(len(keys))
	for i in range(0,len(keys)):
	    
	    p = ax.plot(data_type.get(keys[i],"").flatten(),label=keys[i])
	    ax.axhline(np.mean(data_type.get(keys[i],"").flatten()),label="Average " + keys[i],color=c[i],ls='--',lw=3)
	    ax.legend(loc=2, prop={'size': 14})

