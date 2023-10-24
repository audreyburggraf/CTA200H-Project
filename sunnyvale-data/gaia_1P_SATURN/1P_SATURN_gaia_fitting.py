
# importing packages, functions, etc
# --------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy import cos, sin

import rebound

from scipy.linalg import lstsq
from scipy.optimize import leastsq
from scipy.optimize import minimize
# ------------------------------------------------------------------------------------------------------------------------------

# importing the gaia data 
# ------------------------------------------------------------------------------------------------------------------------------
df = pd.read_csv('gaia_data.csv')
# ------------------------------------------------------------------------------------------------------------------------------



# setting necessary functions
# ------------------------------------------------------------------------------------------------------------------------------
def generate_pm_signal(Delta_alpha, Delta_delta, mu_alpha, mu_delta, times):
    pm_term_ra  = mu_alpha * times  + Delta_alpha
    pm_term_dec = mu_delta * times  + Delta_delta

    return (pm_term_ra, pm_term_dec)
# ------------------------------------------------------------------------------------------------------------------------------
def calculate_thiele_innes(omega, Omega, cos_i, parallax, m_planet, P_orb, m_star=1, G_const=4*np.pi**2):
    n = 2*np.pi/P_orb

    a_AU = (G_const*(m_star+m_planet)/n**2)**(1/3)

    a_hat = (m_planet/(m_star+m_planet))*a_AU*parallax

    sin_i = np.sqrt(1-cos_i**2)

    A = a_hat * (  cos(omega)  * cos(Omega) - sin(omega) * sin(Omega) * cos_i)  
    F = a_hat * ( - sin(omega) * cos(Omega) - cos(omega) * sin(Omega) * cos_i)

    B = a_hat * (  cos(omega)  * sin(Omega) + sin(omega) * cos(Omega) * cos_i)  
    G = a_hat * ( - sin(omega) * sin(Omega) + cos(omega) * cos(Omega) * cos_i)

    C = a_hat * sin(omega)*sin_i
    H = a_hat * sin_i*cos(omega)

    return(B, A, F, G, H, C, a_hat)
# ------------------------------------------------------------------------------------------------------------------------------
def find_var_from_param(e, omega, Omega, cos_i):

    var1 = np.sqrt(cos_i)*cos(Omega)
    var2 = np.sqrt(cos_i)*sin(Omega)
    var3 = np.sqrt(e)*cos(omega)
    var4 = np.sqrt(e)*sin(omega)

    return(var1, var2, var3, var4)
# ------------------------------------------------------------------------------------------------------------------------------
def generate_planet_signal(alpha0, delta0, parallax, var1, var2, var3, var4, m_planet, P_orb, t_peri, times):
    sind, cosd, sina, cosa  = sin(delta0), cos(delta0), sin(alpha0), cos(alpha0)

    e, omega, Omega, cos_i = find_param_from_var(var1, var2, var3, var4)

    B, A, F, G, H, C, _ = calculate_thiele_innes(omega, Omega, cos_i, parallax, m_planet, P_orb)

    M = (2*np.pi)*(times - t_peri)/P_orb  

    E = np.vectorize(rebound.M_to_E)(e,M)

    X = (cos(E)-e)
    Y = np.sqrt((1-e**2))*sin(E)

    DELTA_X = A*X+F*Y            
    DELTA_Y = B*X+G*Y            
    DELTA_Z = H*X+C*Y  

    planetary_ra  = (1/cosd) * (sina*DELTA_X-cosa*DELTA_Y)
    planetary_dec = (-cosd*DELTA_Z + sind*(DELTA_X*cosa+DELTA_Y*sina))

    return (planetary_ra, planetary_dec)
# --------------------------------------------------------------------------------------------------------------------------------
def signal_func_np(pars, alpha0, delta0, times, theta, a_earth = 1):
   
    Delta_alpha, Delta_delta, mu_alpha, mu_delta, parallax = pars

    prop_mot_ra, prop_mot_dec = generate_pm_signal(Delta_alpha, Delta_delta, mu_alpha, mu_delta, times)

    parallax_ra, parallax_dec = generate_parallax_signal(alpha0, delta0, parallax, times)
   
    signal_ra  = prop_mot_ra  + parallax_ra  
    signal_dec = prop_mot_dec + parallax_dec

    eta = signal_ra*cos(theta) + signal_dec*sin(theta)
 
    return(signal_ra, signal_dec, eta)
# ------------------------------------------------------------------------------------------------------------------------------
def signal_func(pars, alpha0, delta0, times, theta, a_earth = 1):
    Delta_alpha, Delta_delta, mu_alpha, mu_delta, parallax, var1, var2, var3, var4, m_planet, P_orb, t_peri = pars
   
    np_pars = Delta_alpha, Delta_delta, mu_alpha, mu_delta, parallax
    signal_ra_np, signal_dec_np, eta_np = signal_func_np(np_pars, alpha0, delta0, times, theta)
   
    planetary_ra, planetary_dec = generate_planet_signal(alpha0, delta0, parallax, var1, var2, var3, var4, m_planet, P_orb, t_peri, times)
   
    signal_ra  = signal_ra_np  + planetary_ra  
    signal_dec = signal_dec_np + planetary_dec

    eta = signal_ra*cos(theta) + signal_dec*sin(theta)

    return(signal_ra, signal_dec, eta)
# ------------------------------------------------------------------------------------------------------------------------------
def calculate_a_hat(parallax, m_planet, P_orb, m_star=1, G_const=4*np.pi**2):
    n = 2*np.pi/P_orb

    a_AU = (G_const*(m_star+m_planet)/n**2)**(1/3)

    a_hat = (m_planet/(m_star+m_planet))*a_AU*parallax

    return(a_hat)
# ------------------------------------------------------------------------------------------------------------------------------
def generate_parallax_signal(alpha0, delta0, parallax, times, a_earth =1):
    d = a_earth/parallax
    sind, cosd, sina, cosa  = sin(delta0), cos(delta0), sin(alpha0), cos(alpha0)

    T = times

    parallax_ra  = (a_earth/(d*cosd))*(sina*cos(2*np.pi*T)-cosa*sin(2*np.pi*T))               #    [rad]
    parallax_dec = (a_earth/d)*sind*cos(2*np.pi*T-alpha0)                                     #    [rad]
       
    return(parallax_ra, parallax_dec)
# ------------------------------------------------------------------------------------------------------------------------------
def find_param_from_var(var1, var2, var3, var4):
       
        omega = np.arctan2(var4,var3)

        Omega = np.arctan2(var2,var1)

        cos_i = var1**2 + var2**2

        e = var3**2 + var4**2
       
        return(e, omega, Omega, cos_i)
# ------------------------------------------------------------------------------------------------------------------------------
def normalized_residuals(pars, alpha0, delta0, sigma, eta_obs, times, theta):
    ra_pred, dec_pred, eta_pred = signal_func(pars,alpha0,delta0, times, theta)
   
    # d_ra  = ra_obs  - ra_pred
    # d_dec = dec_obs - dec_pred

    d_eta = eta_obs - eta_pred

    # return np.concatenate((d_ra, d_dec))
    return d_eta/sigma
# ------------------------------------------------------------------------------------------------------------------------------
def find_chi_squared(y_obs, y_exp, error):
    chi = (y_obs-y_exp)**2/error**2

    chi_squared = np.sum(chi)

    return(chi_squared)
# ------------------------------------------------------------------------------------------------------------------------------



# the main function
# ------------------------------------------------------------------------------------------------------------------------------
def big_func(N, S, times, theta, P_orb):

    # PICK SOME RANDOM NUMBER TO USE FOR GAIA
    x = np.random.randint(0, len(df))
    print("x = ", x)

    # alpha0 and delta0 are from gaia () 
    alpha0 = df.ra[x]*np.pi/180
    delta0 = df.dec[x]*np.pi/180


    mas_rad=1/206264806
    
    inc = [0.022759093, 0.043388885]    # inclination in radians: (Jupiter, Saturn)

    # setting extra parameters needed for truepars
    extra_params = np.zeros((S, 4))

    extra_params[:,0] = 0.1                      # e
    extra_params[:,1] = 5.8674129728             # omega
    extra_params[:,2] = 1.98470185703            # Omega
    extra_params[:,3] = np.cos(np.mean(inc))     # cos i: cos of mean of Jupiter and Saturns inc

    # setting truepars
    truepars = np.zeros((S, 12))

    truepars[:,0]  = np.random.normal(0, 1.93925472e-10, S)    # Delta_alpha_0
    truepars[:,1]  = np.random.normal(0, 1.93925472e-10, S)    # Delta_delta_0
    truepars[:,2]  = df.pmra[x]*mas_rad                        # mu_alpha
    truepars[:,3]  = df.pmdec[x]*mas_rad                       # mu_delta
    truepars[:,4]  = df.parallax[x]*mas_rad                    # parallax
    truepars[:,9]  = 0.0002857                                 # m_planet (Saturn is approx 1/3 times Jupiters mass)
    truepars[:,10] = P_orb*29.4/11.86                          # P_orb
    truepars[:,11] = np.random.uniform(0, truepars[:,10], S)   # t_peri

    for i in range(S):
        truepars[i,5], truepars[i,6], truepars[i,7], truepars[i,8] = find_var_from_param(extra_params[i,0], extra_params[i,1], extra_params[i,2], extra_params[i,3]) # var1, var2, var3, var4


    # calculate a_hat from user true data
    a_hat = calculate_a_hat(truepars[0,4], truepars[0,9], truepars[0,10])

    # create array of errors
    sigma_err = np.geomspace(0.1, 1, S)*a_hat

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
            x = np_best_fit_val[k] # x is equal to kth row of np_best_fit_val
            for j in range(5):
                array[i,j] = M[i,j]*x[j]

        array_row_sums = np.sum(array, axis=1)  
        np_chi_sq[k] = np.sum((array_row_sums - eta_obs[k])**2/sigma_err[k]**2)
    # ----------------------------------------------------------------------------------------------------------


    # ----------------------------- W I T H - P L A N E T - F I T ----------------------------------------------
    # creating arrays
    guess = np.zeros((S,12))
    fitted_params_1P = np.zeros((S, 12))
    eta_best = np.zeros((S, N))
    wp_chi_sq = np.zeros((S))
    
    # bounds 
    bounds = ((-np.inf, np.inf),              # Delta alpha_0 
              (-np.inf,np.inf),               # Delta delta_0 
              (-np.inf,np.inf),               # mu_alpha 
              (-np.inf,np.inf),               # mu_delta 
              (0, np.inf),                    # varpi 
              (-1, 1),                        # var1  
              (-1, 1),                        # var 2
              (-np.sqrt(0.7), np.sqrt(0.7)),  # var3  
              (-np.sqrt(0.7), np.sqrt(0.7)),  # var 4 
              (0,1),                          # m_planet
              (-np.inf, np.inf),              # P_orb
              (-np.inf, np.inf))              # t_peri
    
    # dividing up the parameters 
    guess[:,:] = truepars[:,:12]

    # data where error varys
    for s in range(S):
        # getting best/fitted values 
        fitted_params_1P[s], _, _, _, _ = leastsq(normalized_residuals, guess[s], args=(alpha0, delta0, sigma_err[s], eta_obs[s], times, theta), full_output=1)

        # if nan values then use other method 
        fn = lambda x: normalized_residuals(x,*args) @ normalized_residuals(x,*args)
        args = (alpha0, delta0, sigma_err[s], eta_obs[s], times, theta)

        if np.isnan(fitted_params_1P[s]).any():
            result_min_P1 = minimize(fn, guess[s], method='Nelder-Mead', bounds=bounds)
            fitted_params_1P[s] = result_min_P1.x

        # creating best signal from the best fit
        _, _, eta_best[s] = signal_func(fitted_params_1P[s], alpha0, delta0, times, theta)

        # finding chi squared for with and without planet
        wp_chi_sq[s] = find_chi_squared(eta_best[s], eta_obs[s], sigma_err[s])
        
     
    # ----------------------------------------------------------------------------------------------------------

    # ---------------------------- C A L C U L A T I N G - B I C - A N D - D E L T A - B I C---------------------
    np_BIC = np_chi_sq + 5 * np.log(N)
    wp_BIC = wp_chi_sq + 12*np.log(N)
    Delta_BIC = wp_BIC - np_BIC
    
    print("np_chi_sq = ", np_chi_sq)
    print("wp_chi_sq = ", wp_chi_sq)
    print("np_BIC = ", np_BIC)
    print("wp_BIC = ", wp_BIC)
    print("Delta_BIC = ", Delta_BIC)
    # ----------------------------------------------------------------------------------------------------------
    return(a_hat, sigma_err, Delta_BIC)
# ------------------------------------------------------------------------------------------------------------------------------

import sys
j = int(sys.argv[1])
print('j/100 = ', j)


# initial inputs
# ------------------------------------------------------------------------------------------------------------------------------

N = 100 # number of timesteps
S = 50  # number of errors
K = 10   # number of periods

times = np.linspace(0, 5, N)
theta = np.linspace(0, 2*np.pi, N)

P_orb_array = np.geomspace(0.1, 10, K)


print("P_orb_array = ", P_orb_array)
print("N = ", N, "S = ", S, "K = ", K)

# ------------------------------------------------------------------------------------------------------------------------------

# creating empty arrays
# ------------------------------------------------------------------------------------------------------------------------------
a_hat     = np.zeros((K))
sigma_err = np.zeros((K,S))
Delta_BIC = np.zeros((K,S))
# ------------------------------------------------------------------------------------------------------------------------------

# looping over sigma, P and bin
# --------------------------------------------------------------------------------------------------------------------------------
for k in range(K):
    a_hat[k], sigma_err[k], Delta_BIC[k] = big_func(N, S, times, theta, P_orb_array[k])
# ------------------------------------------------------------------------------------------------------------------------------

np.savez(str(j)+'_gaia_1P_SATURN_data.npz', a_hat, sigma_err, Delta_BIC)



