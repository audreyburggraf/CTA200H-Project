import numpy as np 
from numpy import cos, sin
import rebound 
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------------
# ---------------------------------- SIGNAL COMPONENT EQUATIONS ----------------------------------------------
# ------------------------------------------------------------------------------------------------------------
def generate_pm_signal(Delta_alpha, Delta_delta, mu_alpha, mu_delta, times):
    pm_term_ra  = mu_alpha * times  + Delta_alpha
    pm_term_dec = mu_delta * times  + Delta_delta

    return (pm_term_ra, pm_term_dec)
# ------------------------------------------------------------------------------------------------------------
def generate_parallax_signal(alpha0, delta0, parallax, times, a_earth =1):
    d = a_earth/parallax
    sind, cosd, sina, cosa  = sin(delta0), cos(delta0), sin(alpha0), cos(alpha0)

    T = times 

    parallax_ra  = (a_earth/(d*cosd))*(sina*cos(2*np.pi*T)-cosa*sin(2*np.pi*T))               #    [rad]
    parallax_dec = (a_earth/d)*sind*cos(2*np.pi*T-alpha0)                                     #    [rad]
        
    return(parallax_ra, parallax_dec)
# ------------------------------------------------------------------------------------------------------------
def calculate_thiele_innes(omega, Omega, cos_i, parallax, m_planet, P_orb, m_star=1, G_const=4*np.pi**2):
    n = 2*np.pi/P_orb

    a_AU = (G_const*(m_star+m_planet)/n**2)**(1/3)

    a_hat = (m_planet/(m_star+m_planet))*a_AU*parallax

    # if 1-cos_i**2<0:
        # print("error: square rooting 1-cos(i)^2<0, 1-cos(i)**2 = ", 1-cos_i**2)

    sin_i = np.sqrt(1-cos_i**2)

    A = a_hat * (  cos(omega)  * cos(Omega) - sin(omega) * sin(Omega) * cos_i)  
    F = a_hat * ( - sin(omega) * cos(Omega) - cos(omega) * sin(Omega) * cos_i) 

    B = a_hat * (  cos(omega)  * sin(Omega) + sin(omega) * cos(Omega) * cos_i)  
    G = a_hat * ( - sin(omega) * sin(Omega) + cos(omega) * cos(Omega) * cos_i) 

    C = a_hat * sin(omega)*sin_i
    H = a_hat * sin_i*cos(omega)

    return(B, A, F, G, H, C, a_hat)
# ------------------------------------------------------------------------------------------------------------
def calculate_a_hat(parallax, m_planet, P_orb, m_star=1, G_const=4*np.pi**2):
    n = 2*np.pi/P_orb

    a_AU = (G_const*(m_star+m_planet)/n**2)**(1/3)

    a_hat = (m_planet/(m_star+m_planet))*a_AU*parallax

    return(a_hat)
# ------------------------------------------------------------------------------------------------------------
def find_param_from_var(var1, var2, var3, var4):
        
        omega = np.arctan2(var4,var3)

        Omega = np.arctan2(var2,var1)

        cos_i = var1**2 + var2**2

        e = var3**2 + var4**2
        
        return(e, omega, Omega, cos_i)
# ------------------------------------------------------------------------------------------------------------
def find_var_from_param(e, omega, Omega, cos_i):

    var1 = np.sqrt(cos_i)*cos(Omega)
    var2 = np.sqrt(cos_i)*sin(Omega)
    var3 = np.sqrt(e)*cos(omega)
    var4 = np.sqrt(e)*sin(omega)

    return(var1, var2, var3, var4)
# ------------------------------------------------------------------------------------------------------------
def generate_planet_signal(alpha0, delta0, parallax, var1, var2, var3, var4, m_planet, P_orb, t_peri, times):
    sind, cosd, sina, cosa  = sin(delta0), cos(delta0), sin(alpha0), cos(alpha0)

    e, omega, Omega, cos_i = find_param_from_var(var1, var2, var3, var4)

    B, A, F, G, H, C, _ = calculate_thiele_innes(omega, Omega, cos_i, parallax, m_planet, P_orb)

    M = (2*np.pi)*(times - t_peri)/P_orb  

    E = np.vectorize(rebound.M_to_E)(e,M)
    
    # if 1-e**2<0:
    #     print("error: square rooting 1-e^2<0, 1-e**2 = ", 1-e**2)

    X = (cos(E)-e)
    Y = np.sqrt((1-e**2))*sin(E)

    DELTA_X = A*X+F*Y             
    DELTA_Y = B*X+G*Y             
    DELTA_Z = H*X+C*Y   

    planetary_ra  = (1/cosd) * (sina*DELTA_X-cosa*DELTA_Y) 
    planetary_dec = (-cosd*DELTA_Z + sind*(DELTA_X*cosa+DELTA_Y*sina))

    return (planetary_ra, planetary_dec)
# ------------------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------------------
# ---------------------------------- FULL SIGNAL EQUATIONS ---------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
def signal_func_np_no_eta(pars, alpha0, delta0, times, a_earth = 1):
    
    Delta_alpha, Delta_delta, mu_alpha, mu_delta, parallax = pars

    prop_mot_ra, prop_mot_dec = generate_pm_signal(Delta_alpha, Delta_delta, mu_alpha, mu_delta, times)

    parallax_ra, parallax_dec = generate_parallax_signal(alpha0, delta0, parallax, times)
    
    signal_ra  = prop_mot_ra  + parallax_ra  
    signal_dec = prop_mot_dec + parallax_dec 
 
    return(signal_ra, signal_dec)
# ------------------------------------------------------------------------------------------------------------
def signal_func_np(pars, alpha0, delta0, times, theta, a_earth = 1):
    
    Delta_alpha, Delta_delta, mu_alpha, mu_delta, parallax = pars

    prop_mot_ra, prop_mot_dec = generate_pm_signal(Delta_alpha, Delta_delta, mu_alpha, mu_delta, times)

    parallax_ra, parallax_dec = generate_parallax_signal(alpha0, delta0, parallax, times)
    
    signal_ra  = prop_mot_ra  + parallax_ra  
    signal_dec = prop_mot_dec + parallax_dec 

    eta = signal_ra*cos(theta) + signal_dec*sin(theta)
 
    return(signal_ra, signal_dec, eta)
# ------------------------------------------------------------------------------------------------------------
def signal_func(pars, alpha0, delta0, times, theta, a_earth = 1):
    Delta_alpha, Delta_delta, mu_alpha, mu_delta, parallax, var1, var2, var3, var4, m_planet, P_orb, t_peri = pars
    
    np_pars = Delta_alpha, Delta_delta, mu_alpha, mu_delta, parallax
    signal_ra_np, signal_dec_np, eta_np = signal_func_np(np_pars, alpha0, delta0, times, theta)
    
    planetary_ra, planetary_dec = generate_planet_signal(alpha0, delta0, parallax, var1, var2, var3, var4, m_planet, P_orb, t_peri, times)
    
    signal_ra  = signal_ra_np  + planetary_ra  
    signal_dec = signal_dec_np + planetary_dec 

    eta = signal_ra*cos(theta) + signal_dec*sin(theta)

    return(signal_ra, signal_dec, eta)
# ------------------------------------------------------------------------------------------------------------
def signal_func_n_planets(num_planets, pars, alpha0, delta0, times, theta, a_earth = 1):
    #no planet parameters
    Delta_alpha, Delta_delta, mu_alpha, mu_delta, parallax = pars[0],  pars[1],  pars[2], pars[3], pars[4]
    np_pars = Delta_alpha, Delta_delta, mu_alpha, mu_delta, parallax

    
    # no planet signal
    no_planet_ra, no_planet_dec, no_planet_eta = signal_func_np(np_pars, alpha0, delta0, times, theta)
    
    # planet signal
    planet_signal = np.zeros((2*num_planets, len(times)))
    planetary_ra, planetary_dec = np.zeros((len(times), num_planets)), np.zeros((len(times), num_planets))
    planet_params = np.zeros((num_planets,7))
    planetary_ra_total, planetary_dec_total = np.zeros((1, len(times))), np.zeros((1,len(times)))
    

    for i in range(num_planets):
        planet_params[i] = pars[5+i*7],pars[6+i*7], pars[7+i*7], pars[8+i*7], pars[9+i*7], pars[10+i*7], pars[11+i*7]

        planet_signal[2*i], planet_signal[2*i+1] = generate_planet_signal(alpha0, delta0, parallax, *planet_params[i], times)

        planetary_ra[:,i] = planet_signal[2*i]
        planetary_dec[:,i] = planet_signal[2*i+1]

    planetary_ra_total  = planetary_ra.sum(axis = 1)
    planetary_dec_total = planetary_dec.sum(axis = 1)

    
    # adding no planet and planet signal   
    signal_ra  = no_planet_ra  + planetary_ra_total  
    signal_dec = no_planet_dec + planetary_dec_total 

    eta = signal_ra*cos(theta) + signal_dec*sin(theta)
    
    return(signal_ra, signal_dec, eta, no_planet_ra, no_planet_dec, planetary_ra_total, planetary_dec_total)
# ------------------------------------------------------------------------------------------------------------





# FITTING EQUATIONS
# ------------------------------------------------------------------------------------------------------------
def normalized_residuals(pars, alpha0, delta0, sigma, eta_obs, times, theta):
    ra_pred, dec_pred, eta_pred = signal_func(pars,alpha0,delta0, times, theta)
    
    # d_ra  = ra_obs  - ra_pred
    # d_dec = dec_obs - dec_pred

    d_eta = eta_obs - eta_pred

    # return np.concatenate((d_ra, d_dec))
    return d_eta/sigma
# ------------------------------------------------------------------------------------------------------------
def log_likelihood(pars, alpha0, delta0, sigma, eta_obs, times, theta):
    
    values_list = normalized_residuals(pars, alpha0, delta0, sigma, eta_obs, times, theta)
    
    return -0.5*(values_list @ values_list)
# ------------------------------------------------------------------------------------------------------------
def log_prior(pars):
    alpha, delta, mu_alpha, mu_delta, parallax, var1, var2, var3, var4, m_planet, P_orb, t_peri = pars 
    
    e, omega, Omega, cos_i = find_param_from_var(var1, var2, var3, var4)
    
    if not 0 <= parallax:
        return -np.inf
        
    if not 0 < e < 1:
        return -np.inf
    
    if not 0 < m_planet < 1:
        return -np.inf
    
    if not 0 < P_orb:
        return -np.inf
    
    if not 0 < cos_i < 1:
        return -np.inf
    
    if not -0.5 * P_orb < t_peri <= 0.5 * P_orb:
        return -np.inf
    
    return 0.0
# ------------------------------------------------------------------------------------------------------------
def log_probability(pars, alpha0, delta0, sigma, eta_obs, times, theta):
    lp = log_prior(pars)
    
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + log_likelihood(pars, alpha0, delta0, sigma, eta_obs, times, theta)
# ------------------------------------------------------------------------------------------------------------
def find_chi_squared(y_obs, y_exp, error):
    chi = (y_obs-y_exp)**2/error**2

    chi_squared = np.sum(chi)

    return(chi_squared)




 # ------------------------------------------------------------------------------------------------------------  
def normalized_residuals_np_no_eta(pars, alpha0, delta0, sigma, ra_obs, dec_obs, times):

    ra_pred, dec_pred, = signal_func_np_no_eta(pars, alpha0, delta0, times)
    
    d_ra  = ra_obs  - ra_pred 
    d_dec = dec_obs - dec_pred 
    
    return np.concatenate((d_ra/sigma, d_dec/sigma))
 # ------------------------------------------------------------------------------------------------------------  
def normalized_residuals_np(pars, alpha0, delta0, sigma, ra_obs, dec_obs, times, theta):

    ra_pred, dec_pred, _ = signal_func_np(pars, alpha0, delta0, times, theta)
    
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


    