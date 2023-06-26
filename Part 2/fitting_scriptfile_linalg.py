# importing packages, functions, plot preferences, etc 
# ------------------------------------------------------------------------------------------------------------
import numpy as np

from numpy import cos, sin
from scipy.linalg import lstsq

from functions_new_parameters import signal_func
from functions_new_parameters import normalized_residuals

np.random.seed(37)
# ------------------------------------------------------------------------------------------------------------


# user input 
# ----------------------------------------------------------------------------------------------------------
# enter the true parameters
alpha0, delta0 = 1, 0.3

truepars = np.array((0,                      # alpha                                     [rad]
                     0,                      # delta                                     [rad]
                     2.3084641853871365e-07, # proper motion in RA direction  (mu alpha) [rad/year]
                     1.770935480191023e-07,  # proper motion in Dec direction (mu delta) [rad/year]
                     9.699321049402031e-08,  # parallax                                  [rad]
                     0.5348901624946122,     # var1: sqrt(cosi)cos(Omega)                [unitless]
                     0.8330420709110249,     # var2: sqrt(cosi)sin(Omega)                [unitless]
                     -0.18610652302818084,   # var3: sqrt(e)cos(omega)                   [unitless]
                     0.406650171629573,      # var4: sqrt(e)sin(omega)                   [unitless]
                     0.6,                    # semi major axis                           [AU]
                     0.46146592515998475,    # orbital period                            [years]
                     0.0))                   # time of pericentre passage                [years]

# number of steps for MCMC
step_number = 500

# timescale
times = np.linspace(0, 4.2, 5)
# ------------------------------------------------------------------------------------------------------------

# create true and observed data 
# ------------------------------------------------------------------------------------------------------------
pm_ra_true, prlx_ra_true, true_plnt_ra, true_ra, pm_dec_true, prlx_dec_true, true_plnt_dec, true_dec = signal_func(truepars,alpha0,delta0, times)
result = normalized_residuals(truepars, alpha0, delta0, 0.001, true_ra, true_dec, times)
sigma_err = (1e-5*np.pi/180/60/60)*5

ra_obs  = true_ra   +  np.random.normal(0, sigma_err, len(true_ra)) - alpha0
dec_obs = true_dec  +  np.random.normal(0, sigma_err, len(true_dec))  - delta0
result  = normalized_residuals(truepars, alpha0, delta0, sigma_err, ra_obs, dec_obs, times)
# ------------------------------------------------------------------------------------------------------------


def generate_parallax_signal(alpha, delta, times, a_earth =1):
    d = np.sqrt(alpha**2+delta**2)
        
    sind, cosd, sina, cosa  = sin(delta), cos(delta), sin(alpha), cos(alpha)

    T = times 

    #parallax_dec = (a_earth/d)*sind*cos(2*np.pi*T-alpha)                                     #    [rad]
    parallax_ra  = (a_earth/(d*cosd))*(sina*cos(2*np.pi*T)-cosa*sin(2*np.pi*T))              #    [rad]
        
    return(parallax_ra)

# A_alpha matrix 
# ------------------------------------------------------------------------------------------------------------
A_alpha    = np.zeros((3,len(times))) 
A_alpha[0] = np.ones(len(times))
A_alpha[1] = times
A_alpha[2] = generate_parallax_signal(alpha0, delta0, times)

# ------------------------------------------------------------------------------------------------------------

#lstsq
# ------------------------------------------------------------------------------------------------------------
linalg_test = lstsq(A_alpha.T, ra_obs)
name = ['x', 'residues', 'rank', 's']
#
for i in range(len(linalg_test)):
    print(name[i],(8-len(name[i]))*' ','=',linalg_test[i])
# ------------------------------------------------------------------------------------------------------------

