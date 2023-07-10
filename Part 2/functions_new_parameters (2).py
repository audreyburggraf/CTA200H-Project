import numpy as np 
from numpy import cos, sin
import rebound 
import matplotlib.pyplot as plt

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
def thiele_innes(omega, Omega, cos_i, parallax, m_planet, P_orb, m_star=1, G_const=4*np.pi**2):
    n = 2*np.pi/P_orb

    a_AU = (G_const*(m_star+m_planet)/n**2)**(1/3)

    a_hat = (m_planet/(m_star+m_planet))*a_AU*parallax

    if 1-cos_i**2<0:
        print("error: square rooting 1-cos(i)^2<0, 1-cos(i)**2 = ", 1-cos_i**2)

    sin_i = np.sqrt(1-cos_i**2)

    A = a_hat * (  cos(omega)  * cos(Omega) - sin(omega) * sin(Omega) * cos_i)  
    F = a_hat * ( - sin(omega) * cos(Omega) - cos(omega) * sin(Omega) * cos_i) 

    B = a_hat * (  cos(omega)  * sin(Omega) + sin(omega) * cos(Omega) * cos_i)  
    G = a_hat * ( - sin(omega) * sin(Omega) + cos(omega) * cos(Omega) * cos_i) 

    C = a_hat * sin(omega)*sin_i
    H = a_hat * sin_i*cos(omega)

    return(B, A, F, G, H, C)
# ------------------------------------------------------------------------------------------------------------
def find_param_from_var(var1, var2, var3, var4):
        
        omega = np.arctan2(var4,var3)

        Omega = np.arctan2(var2,var1)

        cos_i = var1**2 + var2**2

        e = var3**2 + var4**2
        
        return(e, omega, Omega, cos_i)
# ------------------------------------------------------------------------------------------------------------
def generate_planet_signal(alpha, delta, parallax, var1, var2, var3, var4, m_planet, P_orb, t_peri, times):
    sind, cosd, sina, cosa  = sin(delta), cos(delta), sin(alpha), cos(alpha)

    e, omega, Omega, cos_i = find_param_from_var(var1, var2, var3, var4)

    B, A, F, G, H, C = thiele_innes(omega, Omega, cos_i, parallax, m_planet, P_orb)

    M = (2*np.pi)*(times - t_peri)/P_orb  

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

    return (X, Y, planetary_ra, planetary_dec)
# ------------------------------------------------------------------------------------------------------------
def signal_func(pars, alpha0, delta0, times, a_earth = 1):
    
    Delta_alpha, Delta_delta, mu_alpha, mu_delta, parallax, var1, var2, var3, var4, m_planet, P_orb, t_peri = pars
    
    delta = delta0 + Delta_delta
    alpha = alpha0 + Delta_alpha
    prop_mot_ra, prop_mot_dec = generate_pm_signal(alpha, delta, mu_alpha, mu_delta, times)

    parallax_ra, parallax_dec = generate_parallax_signal(alpha0, delta0, parallax, times)
    
    _, _, planetary_ra, planetary_dec = generate_planet_signal(alpha0, delta0, parallax, var1, var2, var3, var4, m_planet, P_orb, t_peri, times)
    
    signal_ra  = prop_mot_ra  + parallax_ra  + planetary_ra - alpha
    signal_dec = prop_mot_dec + parallax_dec + planetary_dec - delta 
    
    return(signal_ra, signal_dec)

# ------------------------------------------------------------------------------------------------------------

# FITTING EQUATIONS
# ------------------------------------------------------------------------------------------------------------
def normalized_residuals(pars, alpha0, delta0, sigma, ra_obs, dec_obs, times):
    ra_pred,dec_pred = signal_func(pars,alpha0,delta0, times)
    
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

    parallax_dec = (a_earth/d)*sind*cos(2*np.pi*T-alpha)                                     
    parallax_ra  = (a_earth/(d*cosd))*(sina*cos(2*np.pi*T)-cosa*sin(2*np.pi*T))
        
    return(parallax_ra, parallax_dec)
 # ------------------------------------------------------------------------------------------------------------  
def signal_func_np(pars, alpha0, delta0, times, a_earth = 1):
    
    Delta_alpha, Delta_delta, mu_alpha, mu_delta, parallax = pars

    prop_mot_ra, prop_mot_dec = generate_pm_signal(Delta_alpha, Delta_delta, mu_alpha, mu_delta, times)

    parallax_ra, parallax_dec = generate_parallax_signal_np(alpha0, delta0, parallax, times)
    
    signal_ra  = prop_mot_ra  + parallax_ra  
    signal_dec = prop_mot_dec + parallax_dec 
 
    return(signal_ra, signal_dec)
 # ------------------------------------------------------------------------------------------------------------  
def normalized_residuals_np(pars, alpha0, delta0, sigma, ra_obs, dec_obs, times):

    ra_pred, dec_pred = signal_func_np(pars, alpha0, delta0, times)
    
    d_ra  = ra_obs  - ra_pred 
    d_dec = dec_obs - dec_pred 
    
    return np.concatenate((d_ra/sigma, d_dec/sigma))
# ------------------------------------------------------------------------------------------------------------
def cov_func(M, sigma, print_cov):
	A = M/sigma 

	cov = np.linalg.inv(np.dot(A.T,A))
	
	if print_cov is True:
		print(cov)

	return(cov)
# ------------------------------------------------------------------------------------------------------------
# def create_param_time_series(params, fitted_params, cov, truepars, step_number, color):
#     steps = np.arange(start=1, stop=step_number+1, step=1)

#     pos = np.random.multivariate_normal(fitted_params, cov, size=step_number)
    
#     fig, axes = plt.subplots(len(fitted_params), figsize=(10, 15), sharex=True)

#     plt.subplots_adjust(left = 0.125, bottom = 0.1, right = 0.9, top = 0.9, wspace = 1.5, hspace = .5)


#     for i in range(len(fitted_params)):
#         ax = axes[i]
#         ax.plot(steps, pos[:,i],  alpha=0.3, color=color)
#         ax.axhline(y=truepars[i], color=color, label="true value")
#         ax.set_xlim(0, len(pos))
#         ax.set_ylabel(params[i])
#         ax.yaxis.set_label_coords(-0.1, 0.5)

#         axes[-1].set_xlabel("step number");

#     plt.show()
# ------------------------------------------------------------------------------------------------------------
def log_likelihood_np(pars, alpha0, delta0, x, y, yerr, times):
    
    values_list = normalized_residuals_np(pars, alpha0, delta0, yerr, x, y, times)
    
    return -0.5*(values_list @ values_list)
# ------------------------------------------------------------------------------------------------------------
def log_prior_np(pars):
    Delta_alpha0, Delta_delta0, mu_alpha, mu_delta, parallax = pars 
    
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


    