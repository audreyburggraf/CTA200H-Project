# importing stuff
import numpy as np
from numpy import cos, sin

from scipy.linalg import lstsq
from scipy.optimize import leastsq

from functions_new_parameters import calculate_a_hat
from functions_new_parameters import signal_func
from functions_new_parameters import generate_parallax_signal
from functions_new_parameters import normalized_residuals
from functions_new_parameters import find_chi_squared

def big_func(N, S, times, theta, alpha0, delta0, truepars, truepars_array):

    # calculate a_hat from user true data 
    a_hat = calculate_a_hat(truepars[4], truepars[9], truepars[10])

    # create array of errors 
    sigma_err_array = np.geomspace(0.01, 1, S)*a_hat

    # create noise 
    noise = np.zeros((S,N))

    for i in range(S):
        noise[i] = np.random.normal(0, sigma_err_array[i], N)

    # create true and observed data 
    eta_true_array = np.zeros((S, N))
    eta_obs_array = np.zeros((S, N))

    for i in range(S):
        # finding true signal with and without planet 
        _, _ , eta_true_array[i] = signal_func(truepars_array[i], alpha0, delta0, times, theta) 
        
        # observed data is true data plus some noise 
        eta_obs_array[i] = eta_true_array[i] + noise[i]


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
    np_best_fit_val_array = np.zeros((S, 5))

    for i in range(S):
        np_best_fit_val_array[i], _, _, _ = lstsq(M, eta_obs_array[i])

    array = np.zeros((N, 5))

    np_chi_sq_array = np.zeros((S))

    for k in range(S):
        x = np_best_fit_val_array[k] # x is equal to the kth row of np_best_fit_val_array
        for i in range(N):
            for j in range(5):
                array[i,j] = M[i,j]*x[j]

        array_row_sums = np.sum(array, axis=1)   
        np_chi_sq_array[k] = np.sum((array_row_sums - eta_obs_array[k])**2/sigma_err_array[k]**2)
    # ----------------------------------------------------------------------------------------------------------


    # ----------------------------- W I T H - P L A N E T - F I T ----------------------------------------------
    # creating arrays
    guess_array = np.zeros((S,12))
    wp_best_fit_val_array = np.zeros((S, 12))
    eta_best_array = np.zeros((S, N))
    SN = np.zeros((S))
    wp_chi_sq_array = np.zeros((S))

    # data where error varys 
    for i in range(S):
        # guess
        guess_array[i] = truepars_array[i] 
        
        # getting best/fitted values 
        wp_best_fit_val_array[i], _, _, _, _ = leastsq(normalized_residuals, guess_array[i], args=(alpha0, delta0, sigma_err_array[i], eta_obs_array[i], times, theta), full_output=1)

        # creating best signal from the best fit 
        _, _, eta_best_array[i] = signal_func(wp_best_fit_val_array[i], alpha0, delta0, times, theta)
        
        # finding S/N for each sample
        SN[i] = a_hat/sigma_err_array[i]
        
        # finding chi squared for with and without planet
        wp_chi_sq_array[i] = find_chi_squared(eta_best_array[i], eta_obs_array[i], sigma_err_array[i])
    # ----------------------------------------------------------------------------------------------------------


    # ---------------------------- C A L C U L A T I N G - B I C - A N D - D E L T A - B I C---------------------
    np_BIC = np_chi_sq_array + 5 * np.log(N)
    wp_BIC = wp_chi_sq_array + 12*np.log(N)
    Delta_BIC = wp_BIC - np_BIC
    # ----------------------------------------------------------------------------------------------------------

    return(a_hat, sigma_err_array, Delta_BIC)