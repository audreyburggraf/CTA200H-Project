# importing packages, functions, plot preferences, etc 
# ------------------------------------------------------------------------------------------------------------
import numpy as np
import emcee
import corner

from scipy.optimize import least_squares
from scipy.optimize import leastsq

from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams["figure.dpi"]  = 100
import seaborn as sns
sns.set_style(rc = {'axes.facecolor': 'whitesmoke'})
plt.rcParams["axes.edgecolor"] = '0'
plt.rcParams["axes.linewidth"] = 0.7

color_palette = sns.color_palette("hls", 8)
color0 = color_palette[5] # darker color (error bars)
color1 = color_palette[4] # lighter color (base line)
color2 = color_palette[7] # dark color (over plotted dashed line)

from functions_new_parameters import signal_func
from functions_new_parameters import normalized_residuals
from functions_new_parameters import log_likelihood
from functions_new_parameters import log_prior
from functions_new_parameters import log_probability
from functions_new_parameters import find_param_from_var

np.random.seed(0)
# ------------------------------------------------------------------------------------------------------------


# unit conversion and name arrays
# ------------------------------------------------------------------------------------------------------------
rad_mas = 206264806

parameters = ['alpha', 'delta', 'mu_alpha', 'mu_delta', 'parallax', 'sqrt(cos(i)cos(Omega)', 'sqrt(cosi)sin(Omega)', 'sqrt(e)cos(omega)', 'sqrt(e)sin(omega)' ,'a_AU', 'P_orb', 't_peri']
planet_parameters = ['e', 'omega', 'Omega', 'cos(i)']
# ------------------------------------------------------------------------------------------------------------

# user input 
# ----------------------------------------------------------------------------------------------------------
# enter the true parameters
truepars = np.array((0.7853981641246757,     # alpha                                     [rad]
                     0.7853981637587035,     # delta                                     [rad]
                     2.3084641853871365e-07, # proper motion in RA direction  (mu alpha) [rad/year]
                     1.770935480191023e-07,  # proper motion in Dec direction (mu delta) [rad/year]
                     9.699321049402031e-08,  # parallax                                  [rad]
                     0.5348901624946122,     # var1: sqrt(cosi)cos(Omega)                [unitless]
                     0.8330420709110249,     # var2: sqrt(cosi)sin(Omega)                [unitless]
                     -0.18610652302818084,   # var3: sqrt(e)cos(omega)                   [unitless]
                     0.406650171629573,      # var4: sqrt(e)sin(omega)                   [unitless]
                     0.6,                    # semi major axis                           [AU]
                     0.46146592515998475 ,   # orbital period                            [years]
                     0.0))                   # time of pericentre passage                [years]

# number of steps for MCMC
step_number = 500

# timescale
times = np.linspace(0, 4.2, 100)

# choosing what you want to print/show (yes = 1, no = 0)
print_LS_values, LS_plot, MCMC_plot, MCMC_true_plot, corner_plot, print_tau, print_flat = 1, 1, 1, 1, 1, 1, 0
# ------------------------------------------------------------------------------------------------------------


# create true and observed data 
# ------------------------------------------------------------------------------------------------------------
pm_ra_true, prlx_ra_true, true_plnt_ra, true_ra, pm_dec_true, prlx_dec_true, true_plnt_dec, true_dec = signal_func(truepars, times)
result = normalized_residuals(truepars, 0.001, true_ra, true_dec, times)
sigma_err = (1e-5*np.pi/180/60/60)*5

ra_obs  = true_ra   +  np.random.normal(0, sigma_err, len(true_ra)) 
dec_obs = true_dec  +  np.random.normal(0, sigma_err, len(true_dec)) 
result  = normalized_residuals(truepars, sigma_err, ra_obs, dec_obs, times)
# ------------------------------------------------------------------------------------------------------------

# least square fit 
# ------------------------------------------------------------------------------------------------------------
#result = normalized_residuals(truepars, sigma_err, ra_obs, dec_obs, times)
guess = truepars * (1 + np.random.uniform(0,0.0001))

best, cov, _ , _ , _ = leastsq(normalized_residuals, guess, args=(sigma_err, ra_obs, dec_obs, times), full_output=1)
#result_LS = least_squares(normalized_residuals, guess, args=(sigma_err, ra_obs, dec_obs)) 
#print("cov:", cov)

result = normalized_residuals(best, sigma_err, ra_obs, dec_obs, times)

_, _, best_plnt_ra, best_ra, _, _, best_plnt_dec, best_dec = signal_func(best, times)
planet_param_best = find_param_from_var(best[5], best[6], best[7], best[8])
# ------------------------------------------------------------------------------------------------------------

# MCMC
# ------------------------------------------------------------------------------------------------------------
pos = np.random.multivariate_normal(best , 0.1**2 * cov, size=32)
#pos = best*(1+1e-4 * np.random.randn(32, 12)) 

nwalkers, ndim = pos.shape

initials_good = np.alltrue(np.isfinite([log_probability(z, true_ra, true_dec, sigma_err, times) for z in pos]))

assert initials_good

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = (true_ra, true_dec, sigma_err, times))

sampler.run_mcmc(pos, step_number, progress = True);
# ------------------------------------------------------------------------------------------------------------

















# printing 

# printing what things we are printing/showing
# ------------------------------------------------------------------------------------------------------------
object_number = print_LS_values, LS_plot, MCMC_plot, MCMC_true_plot, corner_plot, print_tau, print_flat
object_name = ['LS fit values', 'LS fit plot', 'MCMC plot', 'MCMC vs true plot', 'corner plot', 'tau', 'flat samples shape']
object_print = []

# for i in range(len(object_number)):
#     if object_number[i] == 1:
#         object_print.append(object_name[i])
# print("We are outputting:",object_print )
# ------------------------------------------------------------------------------------------------------------


# LS fit values 
# ------------------------------------------------------------------------------------------------------------
if print_LS_values == 0 :
    pass
else:
    print("Least squares estimates:")
    for i in range(0, len(parameters)):
        print(parameters[i],(21-len(parameters[i]))*' ','=', best[i])
        
    for j in range(0, len(planet_parameters)):
        print(planet_parameters[j],(21-len(planet_parameters[j]))*' ','=', planet_param_best[j])
# ------------------------------------------------------------------------------------------------------------

# LS fit plot 
# ------------------------------------------------------------------------------------------------------------
if LS_plot == 0 :
    pass
else:
    print(" ")
    print("Least squares plot:")
    fig, ax = plt.subplots(5, 1, figsize=(12, 25), sharex=False)

    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

    ax[0].errorbar(times, (true_plnt_ra)*rad_mas, yerr=sigma_err*rad_mas, fmt="s", markersize=2,  color=color0, label='yerr')
    ax[0].plot(times, (true_plnt_ra)*rad_mas, lw=3, color=color1, label='true')
    ax[0].plot(times, (best_plnt_ra)*rad_mas, lw=3, color=color2, label='best fit', ls='--')
    ax[0].legend(fontsize=5)
    ax[0].set_xlabel("time", fontsize = 15)
    ax[0].set_ylabel("ra [mas]", fontsize = 15)
    ax[0].set_title("RA Planetary Signal vs Time", fontsize = 20)

    ax[1].errorbar(times, (true_plnt_dec)*rad_mas, yerr=sigma_err*rad_mas, fmt="s", markersize=2, color=color0, label='yerr')
    ax[1].plot(times, (true_plnt_dec)*rad_mas, lw=3, color=color1   , label='true')
    ax[1].plot(times, (best_plnt_dec)*rad_mas, lw=3, color=color2, label='best fit', ls='--', )
    ax[1].legend(fontsize=5)
    ax[1].set_xlabel("time", fontsize = 15)
    ax[1].set_ylabel("dec [mas]", fontsize = 15)
    ax[1].set_title("Dec Planetary Signal vs Time", fontsize = 20)

    ax[2].errorbar(times, (true_ra-true_ra[0])*rad_mas, yerr=sigma_err*rad_mas, fmt="s", markersize=2, color=color0, label='yerr')
    ax[2].plot(times, (true_ra-true_ra[0])*rad_mas, lw=3, color=color1       , label='true')
    ax[2].plot(times, (best_ra-best_ra[0])*rad_mas, lw=3, color=color2, label='best fit', ls='--')
    ax[2].legend(fontsize=5)
    ax[2].set_xlabel("time", fontsize = 15)
    ax[2].set_ylabel("ra [mas]", fontsize = 15)
    ax[2].set_title("RA Signal vs Time", fontsize = 20)

    ax[3].errorbar(times, (true_dec-true_dec[0])*rad_mas, yerr=sigma_err*rad_mas, fmt="s", markersize=2, color=color0, label='yerr')
    ax[3].plot(times, (true_dec-true_dec[0])*rad_mas, lw=3, color=color1   , label='true')
    ax[3].plot(times, (best_dec-best_dec[0])*rad_mas, lw=3, color=color2, label='best fit', ls='--', )
    ax[3].legend(fontsize=5)
    ax[3].set_xlabel("time", fontsize = 15)
    ax[3].set_ylabel("dec [mas]", fontsize = 15)
    ax[3].set_title("Dec Signal vs Time", fontsize = 20)

    ax[4].plot((true_ra-true_ra[0])*rad_mas, (true_dec-true_dec[0])*rad_mas, lw=3, color=color1       , label='true')
    ax[4].plot((true_ra-best_ra[0])*rad_mas, (best_dec-best_dec[0])*rad_mas, lw=3, color=color2, label='best fit', ls='--', )
    ax[4].legend(fontsize=5)
    ax[4].set_xlabel("ra [mas]", fontsize = 15)
    ax[4].set_ylabel("dec [mas]", fontsize = 15)
    ax[4].set_title("Signal", fontsize = 20)

    plt.show()
# ------------------------------------------------------------------------------------------------------------


# MCMC plot 
# ------------------------------------------------------------------------------------------------------------
if MCMC_plot == 0:
    pass
else:
    print("MCMC plot:")
    fontsize  = [10, 10, 10, 10, 10, 7, 7, 7, 7, 10, 10, 10]
    fig, axes = plt.subplots(12, figsize=(10, 20), sharex=True)

    plt.subplots_adjust(left = 0.125, bottom = 0.1, right = 0.9, top = 0.9, wspace = 1.5, hspace = .5)

    samples = sampler.get_chain()

    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(parameters[i], fontsize=fontsize[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");

    plt.show()
# ------------------------------------------------------------------------------------------------------------

# comparing mcmc to true data 
# ------------------------------------------------------------------------------------------------------------
if MCMC_true_plot ==0:
    pass
else: 
    print("MCMC vs true plot:")
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=False)

    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
    ax[0].errorbar(times, (true_plnt_ra)*rad_mas, yerr=sigma_err*rad_mas, fmt="s", markersize=2, color=color0, label='yerr')
    ax[0].plot(times, (true_plnt_ra)*rad_mas, lw=4, color=color1, label='true')

    ax[1].errorbar(times, (true_plnt_dec)*rad_mas, yerr=sigma_err*rad_mas, fmt="s", markersize=2, color=color0, label='yerr')
    ax[1].plot(times, (true_plnt_dec)*rad_mas, lw=4, color=color1       , label='true') 

    for sample in samples[-1]:
        _, _, samp_plnt_ra, samp_ra, _, _, samp_plnt_dec, samp_dec = signal_func(sample, times)
        ax[0].plot(times, (samp_plnt_ra)*rad_mas,  lw=0.3, color=color2, label='sample fit', ls='--')
        ax[1].plot(times, (samp_plnt_dec)*rad_mas, lw=0.3, color=color2, label='sample fit', ls='--', )
   

    ax[0].set_xlabel("time", fontsize = 15)
    ax[0].set_ylabel("ra [mas]", fontsize = 15)
    ax[0].set_title("RA Planetary Signal vs Time", fontsize = 20)

    ax[1].set_xlabel("time", fontsize = 15)
    ax[1].set_ylabel("dec [mas]", fontsize = 15)
    ax[1].set_title("Dec Planetary Signal vs Time", fontsize = 20)

    plt.show()
# ------------------------------------------------------------------------------------------------------------

# flat samples
# ------------------------------------------------------------------------------------------------------------
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

if print_flat == 0:
    pass
else:
    print("flat samples shape =", flat_samples.shape)
# ------------------------------------------------------------------------------------------------------------

# corner plot 
# ------------------------------------------------------------------------------------------------------------
if corner_plot==0:       
    pass
else:
    print("Corner plot:")
    fig = corner.corner(
    flat_samples, labels=parameters, truths=truepars, truth_color=color0);

    plt.show()
# ------------------------------------------------------------------------------------------------------------

# tau
# ------------------------------------------------------------------------------------------------------------
if print_tau == 0:
    pass
else:
    tau = sampler.get_autocorr_time()
    print("tau = ", tau)

# ------------------------------------------------------------------------------------------------------------

