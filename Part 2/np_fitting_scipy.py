# importing packages, functions, plot preferences, etc 
# ------------------------------------------------------------------------------------------------------------
import numpy as np
import emcee 

from numpy import cos, sin
from scipy.linalg import lstsq

from scipy.optimize import leastsq

from functions_new_parameters import signal_func_np
from functions_new_parameters import normalized_residuals_np
from functions_new_parameters import generate_parallax_signal_np
#from functions_new_parameters import log_probability_np

parameters = ['Delta alpha0', 'Delta delta0', 'mu_alpha', 'mu_delta', 'parallax']
params_ra  = ['Delta alpha0', 'mu alpha', 'parallax']
params_dec = ['Delta delta0', 'mu delta', 'parallax']

np.random.seed(0)
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------


# user input 
# ----------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# enter the true parameters
alpha0, delta0 = 1, 0.3

truepars = np.array((0,                       # alpha                                     [rad]
                     0,                       # delta                                     [rad]
                     2.3084641853871365e-07,  # proper motion in RA direction  (mu alpha) [rad/year]
                     1.770935480191023e-07,   # proper motion in Dec direction (mu delta) [rad/year]
                     9.699321049402031e-08))  # parallax                                  [rad]

parallax = 9.699321049402031e-08

# number of steps for MCMC
step_number = 500

# timescale
times = np.linspace(0, 4.2, 200)
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------


# create true and observed data 
# ------------------------------------------------------------------------------------------------------------
sigma_err = (1e-5*np.pi/180/60/60)*5

pm_ra_true, prlx_ra_true, true_ra, pm_dec_true, prlx_dec_true, true_dec = signal_func_np(truepars, alpha0, delta0, times)

ra_obs  = true_ra   +  np.random.normal(0, sigma_err, len(true_ra)) 
print(ra_obs[0])
dec_obs = true_dec  +  np.random.normal(0, sigma_err, len(true_dec)) 
# ------------------------------------------------------------------------------------------------------------

# part 1
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
guess = truepars * (1 + np.random.uniform(0,0.0001))

best, cov, _ , _ , _ = leastsq(normalized_residuals_np, guess, args=(alpha0, delta0, sigma_err, ra_obs, dec_obs, times), full_output=1)

print("Part 1: Using scipy.optimize.leastsq")
for i in range(0, len(parameters)):
    print(parameters[i],(21-len(parameters[i]))*' ','=', best[i])
print(" ")
print("cov:", cov)
# ------------------------------------------------------------------------------------------------------------


# part 2 
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
print(" ")
print("Part 2: Using scipy.linalg.stsq")

# M_alpha matrix 
# ------------------------------------------------------------------------------------------------------------
parallax_ra, parallax_dec = generate_parallax_signal_np(alpha0, delta0, 1, times)

M_alpha    = np.zeros((len(params_ra ),len(times))) 
M_alpha[0] = np.ones(len(times))
M_alpha[1] = times
M_alpha[2] = parallax_ra
M_alpha    = M_alpha.T

M_delta    = np.zeros((len(params_dec),len(times))) 
M_delta[0] = np.ones(len(times))
M_delta[1] = times
M_delta[2] = parallax_dec
M_delta    = M_delta.T
# ------------------------------------------------------------------------------------------------------------

#lstsq
# ------------------------------------------------------------------------------------------------------------
x_ra,  res_ra,  rank_ra,  s_ra  = lstsq(M_alpha, ra_obs - alpha0)
x_dec, res_dec, rank_dec, s_dec = lstsq(M_delta, dec_obs - delta0)
# ------------------------------------------------------------------------------------------------------------
def cov_func(M, sigma, print_cov):
	A = M/sigma 

	cov = np.linalg.inv(np.dot(A.T,A))
	
	if print_cov is True:
		print(cov)

	return(cov)

cov_ls_ra  = cov_func(M_alpha, sigma_err, print_cov = False)
cov_ls_dec = cov_func(M_delta, sigma_err, print_cov = False)


# ------------------------------------------------------------------------------------------------------------

#parameters = ['Delta alpha0', 'Delta delta0', 'mu_alpha', 'mu_delta', 'parallax']
resids_alpha = lambda pars:  M_alpha @ np.array([pars[0],pars[2],pars[4]])-(ra_obs-alpha0)
resids_delta = lambda pars:  M_delta @ np.array([pars[1],pars[3],pars[4]])-(dec_obs-delta0)
resids = lambda pars: np.concatenate((resids_alpha(pars),resids_delta(pars)))
x,cov_x,_,_,_ = leastsq(resids,guess,full_output=1)
import matplotlib.pyplot as plt
plt.plot(times, M_delta @  np.array([x[1],x[3],x[4]]) )
plt.plot(times,(dec_obs-delta0))
plt.show()
print(x)
print(cov_x)


print("Fitting for RA:")
for i in range(len(x_ra)):
    print(params_ra[i],  (13-len(params_ra[i])) *' ','=', x_ra[i])
print(" ")
print(cov_ls_ra)
print(" ")
print(" ")
print("Fitting for Dec:")
for i in range(len(x_dec)):
    print(params_dec[i], (13-len(params_dec[i]))*' ','=', x_dec[i])
print(" ")
print(cov_ls_dec)   

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------------------------------------
# pos_ra = np.random.multivariate_normal(x_ra , 0.1**2 * cov_ls_ra, size=32)
# pos_dec = np.random.multivariate_normal(x_dec , 0.1**2 * cov_ls_dec, size=32)

# nwalkers_ra, ndim_ra = pos_ra.shape
# nwalkers_dec, ndim_dec = pos_dec.shape

# initials_good_ra = np.alltrue(np.isfinite([log_probability_np(z,alpha0, delta0, true_ra, true_dec, sigma_err, times) for z in pos_ra]))
# initials_good_dec = np.alltrue(np.isfinite([log_probability_np(y, alpha0, delta0, true_ra, true_dec, sigma_err, times) for y in pos_dec]))

# assert initials_good_ra
# assert initials_good_dec

# sampler_ra = emcee.EnsembleSampler(nwalkers_ra, ndim_ra, log_probability_np, args = (alpha0, delta0, true_ra, true_dec, sigma_err, times))
# sampler_dec = emcee.EnsembleSampler(nwalkers_dec, ndim_dec, log_probability_np, args = (alpha0, delta0, true_ra, true_dec, sigma_err, times))

# sampler_ra.run_mcmc(pos_ra, step_number, progress = True);
# sampler_dec.run_mcmc(pos_dec, step_number, progress = True);
# # ------------------------------------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------------------------------------


