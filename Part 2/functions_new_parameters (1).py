import numpy as np 
from numpy import cos, sin
import rebound 

# SIGNAL EQUATIONS
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
def generate_pm_signal(alpha, delta, mu_alpha, mu_delta, times):
    pm_term_ra  = mu_alpha * times  + alpha
    pm_term_dec = mu_delta * times  + delta

    return (pm_term_ra, pm_term_dec)
# ------------------------------------------------------------------------------------------------------------
def generate_parallax_signal(alpha, delta, parallax, times, a_earth =1):
    d = 1/parallax
    sind, cosd, sina, cosa  = sin(delta), cos(delta), sin(alpha), cos(alpha)

    T = times 

    parallax_dec = (a_earth/d)*sind*cos(2*np.pi*T-alpha)                                     #    [rad]
    parallax_ra  = (a_earth/(d*cosd))*(sina*cos(2*np.pi*T)-cosa*sin(2*np.pi*T))              #    [rad]
        
    return(parallax_ra, parallax_dec)
# ------------------------------------------------------------------------------------------------------------
def thiele_innes(omega, Omega, cos_i, a_AU, parallax):
    a_hat = 0.014116666278885887*a_AU*parallax

    if 1-cos_i**2<0:
        print("error: square rooting 1-cos(i)^2<0, 1-cos(i)**2 = ", 1-cos_i**2)

    sin_i = np.sqrt(1-cos_i**2)

    A = a_hat * (  cos(omega)  * cos(Omega) - sin(omega) * sin(Omega) * cos_i)  
    F = a_hat * ( - sin(omega) * cos(Omega) - cos(omega) * sin(Omega) * cos_i) 

    B = a_hat * (  cos(omega)  * sin(Omega) + sin(omega) * cos(Omega) * cos_i)  
    G = a_hat * ( - sin(omega) * sin(Omega) + cos(omega) * cos(Omega) * cos_i) 

    C = a_hat * sin(omega)*sin_i
    H = a_hat * sin_i*cos(omega)

    return(A, F, B, G, H, C)
# ------------------------------------------------------------------------------------------------------------
def E_from_M(e,M, tol=1e-10):
    E = M
    f=np.inf
    while np.abs(f) > (tol):
        f = E-e*sin(E)-M
        f_prime = 1-e*cos(E)
        E = E -f/f_prime
    return(E)
# ------------------------------------------------------------------------------------------------------------
def find_param_from_var(var1, var2, var3, var4):
        
        omega = np.arctan2(var4,var3)

        Omega = np.arctan2(var2,var1)

        cos_i = var1**2 + var2**2

        e = var3**2 + var4**2
        
        return(e, omega, Omega, cos_i)
# ------------------------------------------------------------------------------------------------------------
def generate_planet_signal(alpha, delta, var1, var2, var3, var4, a_AU, parallax, P_orb, t_peri, times):
    sind, cosd, sina, cosa  = sin(delta), cos(delta), sin(alpha), cos(alpha)

    e, omega, Omega, cos_i = find_param_from_var(var1, var2, var3, var4)

    A, F, B, G, H, C = thiele_innes(omega, Omega, cos_i, a_AU, parallax)

    M = (2*np.pi)*(times - t_peri)/P_orb  

    # calculate E using the function written
    #E = np.vectorize(E_from_M)(e, M)

    # calculate E using rebound
    E = np.vectorize(rebound.M_to_E)(e,M)
    
    
    if 1-e**2<0:
        print("error: square rooting 1-e^2<0, 1-e**2 = ", 1-e**2)
    X = (cos(E)-e)
    Y = np.sqrt((1-e**2))*sin(E)

    DELTA_X = A*X+F*Y             
    DELTA_Y = B*X+G*Y             
    DELTA_Z = H*X+C*Y   

    planetary_ra  = (1/cosd) * (sina*DELTA_X-cosa*DELTA_Y) 
    planetary_dec = (-cosd*DELTA_Z + sind*(DELTA_X*cosa+DELTA_Y*sina))

    return (planetary_ra, planetary_dec)
# ------------------------------------------------------------------------------------------------------------
def signal_func(pars, alpha0, delta0, times, a_earth = 1):
    
    Delta_alpha, Delta_delta, mu_alpha, mu_delta, parallax, var1, var2, var3, var4, a_AU, P_orb, t_peri = pars
    delta = delta0+Delta_delta
    alpha = alpha0 + Delta_alpha
    prop_mot_ra, prop_mot_dec = generate_pm_signal(alpha, delta, mu_alpha, mu_delta, times)

    parallax_ra, parallax_dec = generate_parallax_signal(alpha0, delta0, parallax, times)
    
    planetary_ra, planetary_dec = generate_planet_signal(alpha0, delta0, var1, var2, var3, var4, a_AU, parallax, P_orb, t_peri, times)
    
    signal_ra  = prop_mot_ra  + parallax_ra  + planetary_ra - alpha
    signal_dec = prop_mot_dec + parallax_dec + planetary_dec - delta 
    
    return(prop_mot_ra, parallax_ra, planetary_ra, signal_ra, prop_mot_dec, parallax_dec, planetary_dec, signal_dec)

# ------------------------------------------------------------------------------------------------------------

# FITTING EQUATIONS
# ------------------------------------------------------------------------------------------------------------
def normalized_residuals(pars, alpha0, delta0, sigma, ra_obs, dec_obs, times):
    _, _, _, ra_pred, _, _, _, dec_pred = signal_func(pars,alpha0,delta0, times)
    
    d_ra  = ra_obs  - ra_pred
    d_dec = dec_obs - dec_pred
    
    return np.concatenate((d_ra/sigma, d_dec/sigma))
# ------------------------------------------------------------------------------------------------------------
def log_likelihood(pars, x, y, yerr, times):
    
    values_list = normalized_residuals(pars, yerr, x, y, times)
    
    return -0.5*(values_list @ values_list)
# ------------------------------------------------------------------------------------------------------------
def log_prior(pars):
    alpha, delta, mu_alpha, mu_delta, parallax, var1, var2, var3, var4, a_AU, P_orb, t_peri = pars 
    
    e, omega, Omega, cos_i = find_param_from_var(var1, var2, var3, var4)
    
    if not 0 <= parallax:
        return -np.inf
        
    if not 0 < e < 1:
        return -np.inf
    
    if not 0 < a_AU:
        return -np.inf
    
    if not 0 < P_orb:
        return -np.inf
    
    if not 0 < cos_i < 1:
        return -np.inf
    
    if not -0.5 * P_orb < t_peri <= 0.5 * P_orb:
        return -np.inf
    
    return 0.0
# ------------------------------------------------------------------------------------------------------------
def log_probability(pars, x, y, yerr, times):
    lp = log_prior(pars)
    
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + log_likelihood(pars, x, y, yerr, times)
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------





# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
def generate_parallax_signal_np(alpha, delta, parallax, times, a_earth =1):
    d = a_earth/parallax

    sind, cosd, sina, cosa  = sin(delta), cos(delta), sin(alpha), cos(alpha)

    T = times 

    parallax_dec = (a_earth/d)*sind*cos(2*np.pi*T-alpha)                                     #    [rad]
    parallax_ra  = (a_earth/(d*cosd))*(sina*cos(2*np.pi*T)-cosa*sin(2*np.pi*T))              #    [rad]
        
    return(parallax_ra, parallax_dec)
 # ------------------------------------------------------------------------------------------------------------  
def signal_func_np(pars, alpha0, delta0, times, a_earth = 1):
    
    Delta_alpha, Delta_delta, mu_alpha, mu_delta, parallax = pars

    delta = delta0 + Delta_delta
    alpha = alpha0 + Delta_alpha

    prop_mot_ra, prop_mot_dec = generate_pm_signal(Delta_alpha, Delta_delta, mu_alpha, mu_delta, times)

    parallax_ra, parallax_dec = generate_parallax_signal_np(alpha0, delta0, parallax, times)
    
    signal_ra  = prop_mot_ra  + parallax_ra  
    signal_dec = prop_mot_dec + parallax_dec 
 
    return(prop_mot_ra, parallax_ra, signal_ra, prop_mot_dec, parallax_dec, signal_dec)
 # ------------------------------------------------------------------------------------------------------------  
def normalized_residuals_np(pars, alpha0, delta0, sigma, ra_obs, dec_obs, times):

    _, _, ra_pred, _, _, dec_pred = signal_func_np(pars, alpha0, delta0, times)
    
    d_ra  = ra_obs  - ra_pred 
    d_dec = dec_obs - dec_pred 
    
    return np.concatenate((d_ra/sigma, d_dec/sigma))
# ------------------------------------------------------------------------------------------------------------
def log_likelihood_np(pars, alpha0, delta0, x, y, yerr, times):
    
    values_list = normalized_residuals_np(pars, alpha0, delta0, yerr, x, y, times)
    
    return -0.5*(values_list @ values_list)
# ------------------------------------------------------------------------------------------------------------
def log_prior_np(pars):
    _, _, parallax= pars 
    
    if not 0 <= parallax:
        return -np.inf
    
    return 0.0
# ------------------------------------------------------------------------------------------------------------
def log_probability_np(pars, alpha0, delta0, x, y, yerr, times):
    lp = log_prior_np(pars)
    
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + log_likelihood_np(pars, alpha0, delta0, x, y, yerr, times)
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------


def cov_func(M, sigma, print_cov):
	A = M/sigma 

	cov = np.linalg.inv(np.dot(A.T,A))
	
	if print_cov is True:
		print(cov)

	return(cov)