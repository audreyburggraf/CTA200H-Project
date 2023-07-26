# importing packages, functions, etc 
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import rebound 

from numpy import cos, sin

from scipy.linalg import lstsq
from scipy.optimize import leastsq



# importing necessary functions 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
def generate_pm_signal(Delta_alpha, Delta_delta, mu_alpha, mu_delta, times):
    pm_term_ra  = mu_alpha * times  + Delta_alpha
    pm_term_dec = mu_delta * times  + Delta_delta

    return (pm_term_ra, pm_term_dec)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
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
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
def find_var_from_param(e, omega, Omega, cos_i):

    var1 = np.sqrt(cos_i)*cos(Omega)
    var2 = np.sqrt(cos_i)*sin(Omega)
    var3 = np.sqrt(e)*cos(omega)
    var4 = np.sqrt(e)*sin(omega)

    return(var1, var2, var3, var4)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
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
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
def signal_func_np(pars, alpha0, delta0, times, theta, a_earth = 1):
    
    Delta_alpha, Delta_delta, mu_alpha, mu_delta, parallax = pars

    prop_mot_ra, prop_mot_dec = generate_pm_signal(Delta_alpha, Delta_delta, mu_alpha, mu_delta, times)

    parallax_ra, parallax_dec = generate_parallax_signal(alpha0, delta0, parallax, times)
    
    signal_ra  = prop_mot_ra  + parallax_ra  
    signal_dec = prop_mot_dec + parallax_dec 

    eta = signal_ra*cos(theta) + signal_dec*sin(theta)
 
    return(signal_ra, signal_dec, eta)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
def signal_func(pars, alpha0, delta0, times, theta, a_earth = 1):
    Delta_alpha, Delta_delta, mu_alpha, mu_delta, parallax, var1, var2, var3, var4, m_planet, P_orb, t_peri = pars
    
    np_pars = Delta_alpha, Delta_delta, mu_alpha, mu_delta, parallax
    signal_ra_np, signal_dec_np, eta_np = signal_func_np(np_pars, alpha0, delta0, times, theta)
    
    planetary_ra, planetary_dec = generate_planet_signal(alpha0, delta0, parallax, var1, var2, var3, var4, m_planet, P_orb, t_peri, times)
    
    signal_ra  = signal_ra_np  + planetary_ra  
    signal_dec = signal_dec_np + planetary_dec 

    eta = signal_ra*cos(theta) + signal_dec*sin(theta)

    return(signal_ra, signal_dec, eta)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
def calculate_a_hat(parallax, m_planet, P_orb, m_star=1, G_const=4*np.pi**2):
    n = 2*np.pi/P_orb

    a_AU = (G_const*(m_star+m_planet)/n**2)**(1/3)

    a_hat = (m_planet/(m_star+m_planet))*a_AU*parallax

    return(a_hat)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
def generate_parallax_signal(alpha0, delta0, parallax, times, a_earth =1):
    d = a_earth/parallax
    sind, cosd, sina, cosa  = sin(delta0), cos(delta0), sin(alpha0), cos(alpha0)

    T = times 

    parallax_ra  = (a_earth/(d*cosd))*(sina*cos(2*np.pi*T)-cosa*sin(2*np.pi*T))               #    [rad]
    parallax_dec = (a_earth/d)*sind*cos(2*np.pi*T-alpha0)                                     #    [rad]
        
    return(parallax_ra, parallax_dec)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
def find_param_from_var(var1, var2, var3, var4):
        
        omega = np.arctan2(var4,var3)

        Omega = np.arctan2(var2,var1)

        cos_i = var1**2 + var2**2

        e = var3**2 + var4**2
        
        return(e, omega, Omega, cos_i)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
def normalized_residuals(pars, alpha0, delta0, sigma, eta_obs, times, theta):
    ra_pred, dec_pred, eta_pred = signal_func(pars,alpha0,delta0, times, theta)
    
    # d_ra  = ra_obs  - ra_pred
    # d_dec = dec_obs - dec_pred

    d_eta = eta_obs - eta_pred

    # return np.concatenate((d_ra, d_dec))
    return d_eta/sigma
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
def find_chi_squared(y_obs, y_exp, error):
    chi = (y_obs-y_exp)**2/error**2

    chi_squared = np.sum(chi)

    return(chi_squared)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
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
    #SN = np.zeros((S))
    wp_chi_sq = np.zeros((S))

    # data where error varys 
    for i in range(S):
        # guess
        guess[i] = truepars[i] 
        
        # getting best/fitted values 
        wp_best_fit_val[i], _, _, _, _ = leastsq(normalized_residuals, guess[i], args=(alpha0, delta0, sigma_err[i], eta_obs[i], times, theta), full_output=1)

        # creating best signal from the best fit 
        _, _, eta_best[i] = signal_func(wp_best_fit_val[i], alpha0, delta0, times, theta)
        
        # finding S/N for each sample
        #SN[i] = a_hat/sigma_err[i]
        
        # finding chi squared for with and without planet
        wp_chi_sq[i] = find_chi_squared(eta_best[i], eta_obs[i], sigma_err[i])
    
    SN = a_hat/sigma_err
    # ----------------------------------------------------------------------------------------------------------

    # ---------------------------- C A L C U L A T I N G - B I C - A N D - D E L T A - B I C---------------------
    np_BIC = np_chi_sq + 5 * np.log(N)
    wp_BIC = wp_chi_sq + 12*np.log(N)
    Delta_BIC = wp_BIC - np_BIC
    # ----------------------------------------------------------------------------------------------------------
    return(a_hat, sigma_err, Delta_BIC, SN)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------






# initial inputs 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
N = 100 # number of timesteps 
S = 10  # number of errors
K = 5   # number of periods
J = 5   # number of entries in each bin

times = np.linspace(0, 5, N)
theta = np.linspace(0, 2*np.pi, N)

alpha0, delta0 = 1, 0.3

# colors = sns.color_palette("hls", J*K)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

# looping over P, sigma and bin
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
P_orb_array = np.linspace(0.1, 10, K)

P_orb_plot_array = np.zeros((K,S))

for k in range(K):
    P_orb_plot_array[k] = P_orb_array[k]


a_hat     = np.zeros((J,K))
sigma_err = np.zeros((J,K,S))
Delta_BIC = np.zeros((J,K,S))
SN        = np.zeros((J,K,S))

for j in range(J):
    for k in range(K):
        a_hat[j,k], sigma_err[j,k], Delta_BIC[j,k], SN[j,k] = big_func(N, S, times, theta, alpha0, delta0, P_orb_array[k])
# ---------------------------------------------------------------------------------------------------------------------------------------------------------


# making detectability data 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
yn_detection = np.zeros((J,K,S))

for j in range(J):
    for k in range(K):
        for s in range(S):
            if -Delta_BIC[j][k][s] > 20:
                yn_detection[j][k][s] = 1
            else:
                yn_detection[j][k][s] = 0

sum = np.zeros((K,S))

for j in range(J):
    sum = sum + yn_detection[j]
    
detection_frac = sum/J
# ---------------------------------------------------------------------------------------------------------------------------------------------------------


# # plotting
# # ---------------------------------------------------------------------------------------------------------------------------------------------------------
# for k in range(K):
#     for j in range(J):
#         # marker = markers[2*k+j]
#         color = colors[2*k+j]
#         plt.scatter(a_hat[j,k]/sigma_err[j,k], -1*Delta_BIC[j,k], label= P_orb_array[k], color=colors[2*k+j])


# plt.xscale('log')
# plt.yscale('log')


# plt.xlabel('S/N', fontsize = 15)
# plt.ylabel('$-\Delta$BIC', fontsize=15)
# plt.title('-$\Delta$BIC vs S/N',fontsize=20)

# plt.axhline(20, color='gray')
# plt.text(5.9, 23, '-$\Delta$BIC = 20', fontsize = 15, color='gray')
# plt.show
# # ---------------------------------------------------------------------------------------------------------------------------------------------------------
# for k in range(K):
#     for j in range(J):
#         plt.scatter(a_hat[j,k]/sigma_err[j,k], detection_frac[k],  label= P_orb_array[k], color=colors[2*k+j])


# # plt.xscale('log')
# # plt.yscale('log')


# plt.axhline(20, color='gray')
# plt.text(5.9, 23, '-$\Delta$BIC = 20', fontsize = 15, color='gray')


# # plot legend
# # legend=plt.legend(loc="lower right",frameon=True, markerscale = 1, bbox_to_anchor=(1.2, 0.4), title="$P_{orb}$ Value")
# # legend.get_frame().set_edgecolor('0.3')
# # legend.get_frame().set_linewidth(1)
# # ---------------------------------------------------------------------------------------------------------------------------------------------------------
# for j in range(J):
#         for k in range(K):
#             plt.hexbin(SN[j,k], P_orb_plot_array[k], C = detection_frac[k], gridsize=5, cmap='YlGn')


# cb = plt.colorbar(shrink=0.9, label="% det.")
# plt.title('Figure 5', fontsize=20)
# plt.xlabel('$S/N$', fontsize=15)
# plt.ylabel('$P_{orb}$ [years]', fontsize=15)


# plt.show
# # ---------------------------------------------------------------------------------------------------------------------------------------------------------