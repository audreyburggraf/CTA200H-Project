# importing stuff
import numpy as np
from numpy import cos, sin

from scipy.linalg import lstsq

from scipy.optimize import leastsq
from scipy.optimize import minimize
from scipy.optimize import Bounds

from functions_new_parameters import find_var_from_param
from functions_new_parameters import calculate_a_hat
from functions_new_parameters import signal_func
from functions_new_parameters import generate_parallax_signal
from functions_new_parameters import normalized_residuals
from functions_new_parameters import find_chi_squared





def big_func(N, S, times, theta, alpha0, delta0, P_orb):

    # setting extra parameters needed for truepars
    extra_params = np.zeros((S, 4))

    extra_params[:,0] = np.random.uniform(0, 0.5, S)      # e
    extra_params[:,1] = np.random.uniform(0, 2*np.pi, S)  # omega
    extra_params[:,2] = np.random.uniform(0, 2*np.pi, S)  # Omega
    extra_params[:,3] = np.random.uniform(0, 1, S)        # cos i 

    # setting truepars 
    truepars = np.zeros((S, 12))

    truepars[:,0]  = np.random.normal(0, 1.93925472e-10, S)         # Delta_alpha_0
    truepars[:,1]  = np.random.normal(0, 1.93925472e-10, S)         # Delta_delta_0
    truepars[:,2]  = 2.3084641853871365e-07                         # mu_alpha
    truepars[:,3]  = 1.770935480191023e-07                          # mu_delta 
    truepars[:,4]  = 9.699321049402031e-08                          # parallax
    truepars[:,9]  = 0.0143188                                      # m_planet 
    truepars[:,10] = P_orb                                          # P_orb 
    truepars[:,11] = np.random.uniform(0, truepars[:,10], S)        # t_peri

    for i in range(S):
        truepars[i,5], truepars[i,6], truepars[i,7], truepars[i,8] = find_var_from_param(extra_params[i,0], extra_params[i,1], extra_params[i,2], extra_params[i,3]) # var1, var2, var3, var4

    # calculate a_hat from user true data 
    a_hat = calculate_a_hat(truepars[0,4], truepars[0,9], truepars[0,10])

    # create array of errors 
    sigma_err = np.geomspace(0.1, 1, S)*a_hat
    # sigma_err = np.geomspace(0.01, 0.1, S)*a_hat

    # create noise 
    noise = np.zeros((S,N))

    for i in range(S):
        noise[i] = np.random.normal(0, sigma_err[i], N)

    # create true and observed data 
    eta_true = np.zeros((S, N))
    eta_obs = np.zeros((S, N))

    for i in range(S):
        # finding true signal with and without planet 
        _, _ , eta_true[i] = signal_func(truepars[i], alpha0, delta0, times, theta) 
        
        # observed data is true data plus some noise 
        eta_obs[i] = eta_true[i] + noise[i]




    # ------------------------------- N O - P L A N E T - F I T --------------------------------------
    PI_ra, PI_dec = generate_parallax_signal(alpha0, delta0, 1, times)

    M = np.zeros((N, 5))

    for i in range(N):
        M[i,0] = cos(theta[i])
        M[i,1] = sin(theta[i])
        M[i,2] = cos(theta[i]) * times[i]
        M[i,3] = sin(theta[i]) * times[i]
        M[i,4] = cos(theta[i])*PI_ra[i] + sin(theta[i])*PI_dec[i]

    # finding the best fit values for the S samples 
    np_best_fit_val = np.zeros((S, 5))

    for i in range(S):
        np_best_fit_val[i], _, _, _ = lstsq(M, eta_obs[i]) 

    array = np.zeros((N, 5))

    np_chi_sq = np.zeros((S))

    for k in range(S):
        for i in range(N):
            x = np_best_fit_val[k] # x is equal tokth row of np_best_fit_val
            for j in range(5):
                array[i,j] = M[i,j]*x[j]

        array_row_sums = np.sum(array, axis=1)   
        np_chi_sq[k] = np.sum((array_row_sums - eta_obs[k])**2/sigma_err[k]**2)
    # ----------------------------------------------------------------------------------------------------------




    # ----------------------------- W I T H - P L A N E T - F I T ----------------------------------------------
    # creating arrays
    guess = np.zeros((S,12))
    wp_best_fit_val = np.zeros((S, 12))
    eta_best = np.zeros((S, N))
    wp_chi_sq = np.zeros((S))

    # bounds for other method
    bounds = ((-np.inf, np.inf),              # Delta alpha_0 
              (-np.inf,np.inf),               # Delta delta_0 
              (-np.inf,np.inf),               # mu_alpha 
              (-np.inf,np.inf),               # mu_delta 
              (0, np.inf),                    # varpi 
              (-1, 1),                        # var3  
              (-1, 1),                        # var 4 
              (-np.sqrt(0.7), np.sqrt(0.7)),  # var3  
              (-np.sqrt(0.7), np.sqrt(0.7)),  # var 4 
              (0,1),                          # m_planet
              (-np.inf, np.inf),              # P_orb
              (-np.inf, np.inf))              # t_peri

    # data where error varys 
    for i in range(S):
        # guess
        guess[i] = truepars[i] 

        # arguments
        args = (alpha0, delta0, sigma_err[i], eta_obs[i], times, theta)
        
        # getting best/fitted values 
        wp_best_fit_val[i], _, _, _, _ = leastsq(normalized_residuals, guess[i], args=(alpha0, delta0, sigma_err[i], eta_obs[i], times, theta), full_output=1)

        # if there are nan values then use other method 
        fn = lambda x: normalized_residuals(x,*args) @ normalized_residuals(x,*args)

        if np.isnan(wp_best_fit_val[i]).any():
            fn = lambda x: normalized_residuals(x,*args) @ normalized_residuals(x,*args)
            result_min = minimize(fn,guess[j],method='Nelder-Mead', bounds=bounds)
            wp_best_fit_val[i] = result_min.x


        # creating best signal from the best fit 
        _, _, eta_best[i] = signal_func(wp_best_fit_val[i], alpha0, delta0, times, theta)
        
        # finding chi squared for with and without planet
        wp_chi_sq[i] = find_chi_squared(eta_best[i], eta_obs[i], sigma_err[i])
    # ----------------------------------------------------------------------------------------------------------



    # ---------------------------- C A L C U L A T I N G - B I C - A N D - D E L T A - B I C---------------------
    np_BIC = np_chi_sq + 5 * np.log(N)
    wp_BIC = wp_chi_sq + 12*np.log(N)
    Delta_BIC = wp_BIC - np_BIC


    # checking for nan value 
    if np.isnan(Delta_BIC).any() == True:
        print('There is a Delta_BIC value that is Nan')
    else:
        print('There are no Nan Delta_BIC values ')
    # ----------------------------------------------------------------------------------------------------------

    return(a_hat, sigma_err, Delta_BIC)