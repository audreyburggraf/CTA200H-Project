import numpy as np 
from numpy import cos, sin, arcsin, arccos, arctan

def generate_pm_signal(alpha, delta, mu_alpha, mu_delta, times):
    
    pm_term_ra  = mu_alpha * times   + alpha
    pm_term_dec = mu_delta * times  + delta

    return (pm_term_ra, pm_term_dec)


def generate_parallax_signal(alpha, delta, parallax, times, a_earth =1):
	d = 1/parallax
        
	sind, cosd, sina, cosa  = sin(delta), cos(delta), sin(alpha), cos(alpha)

	T = times 

	parallax_dec = (a_earth/d)*sind*cos(2*np.pi*T-alpha)                                     #    [rad]
	parallax_ra  = (a_earth/(d*cosd))*(sina*cos(2*np.pi*T)-cosa*sin(2*np.pi*T))              #    [rad]
      
	return(parallax_ra, parallax_dec)


def thiele_innes(omega, Omega, inc, a_AU, parallax):
    a_hat = 0.014116666278885887*a_AU*parallax

    A = a_hat * (  cos(omega)  * cos(Omega) - sin(omega) * sin(Omega) * cos(inc))  
    F = a_hat * ( - sin(omega) * cos(Omega) - cos(omega) * sin(Omega) * cos(inc)) 

    B = a_hat * (  cos(omega)  * sin(Omega) + sin(omega) * cos(Omega) * cos(inc))  
    G = a_hat * ( - sin(omega) * sin(Omega) + cos(omega) * cos(Omega) * cos(inc)) 

    C = a_hat * sin(omega)*sin(inc)
    H = a_hat * sin(inc)*cos(omega)

    return(A, F, B, G, H, C)

def E_from_M(e,M, tol=1e-10):
	E = M
	f=np.inf
	while np.abs(f) > (tol):
		f = E-e*sin(E)-M
		f_prime = 1-e*cos(E)
		E = E -f/f_prime
	return(E)

def generate_planet_signal(alpha, delta, omega, Omega, inc, a_AU, parallax, P_orb, t_peri, e, times):
    sind, cosd, sina, cosa  = sin(delta), cos(delta), sin(alpha), cos(alpha)

    A, F, B, G, H, C = thiele_innes(omega, Omega, inc, a_AU, parallax)

    M = (2*np.pi)*(times - t_peri)/P_orb  
    E = np.vectorize(E_from_M)(e, M)

    X = (cos(E)-e)   

    if 1-e**2 < 0:
        print("1-e^2 = ", 1-e**2)

    Y = np.sqrt((1-e**2))*sin(E)

    DELTA_X = A*X+F*Y             
    DELTA_Y = B*X+G*Y             
    DELTA_Z = H*X+C*Y   

    planetary_ra  = (1/cosd) * (sina*DELTA_X-cosa*DELTA_Y) 
    planetary_dec = (-cosd*DELTA_Z + sind*(DELTA_X*cosa+DELTA_Y*sina))

    return (planetary_ra, planetary_dec)


def generate_astrometry_signal(alpha, delta, parallax, mu_alpha, mu_delta, omega, Omega, inc, e, a_AU, P_orb, t_peri, times, a_earth = 1):

    a_hat = 0.014116666278885887*a_AU*parallax  # a_hat is not correct 
    
    prop_mot_ra, prop_mot_dec = generate_pm_signal(alpha, delta, mu_alpha, mu_delta, times)
    
    parallax_ra, parallax_dec = generate_parallax_signal(alpha, delta, parallax, times)
    
    planetary_ra, planetary_dec = generate_planet_signal(alpha, delta, omega, Omega, inc, a_AU, parallax, P_orb, t_peri, e, times)
    
    signal_ra  = prop_mot_ra  + parallax_ra  + planetary_ra
    signal_dec = prop_mot_dec + parallax_dec + planetary_dec
    
    return(signal_ra, signal_dec, prop_mot_ra, prop_mot_dec, parallax_ra, parallax_dec, planetary_ra, planetary_dec)

def generate_astrometry_signal_func(alpha, delta, parallax, mu_alpha, mu_delta, omega, Omega, inc, e, a_AU, P_orb, t_peri, times, a_earth = 1):

    a_hat = 0.014116666278885887*a_AU*parallax  # a_hat is not correct 
    
    prop_mot_ra, prop_mot_dec = generate_pm_signal(mu_alpha, mu_delta, times)
    
    parallax_ra, parallax_dec = generate_parallax_signal(alpha, delta, parallax, times)
    
    planetary_ra, planetary_dec = generate_planet_signal(alpha, delta, omega, Omega, inc, a_AU, parallax, P_orb, t_peri, e, times)
    
    signal_ra  = prop_mot_ra  + parallax_ra  + planetary_ra
    signal_dec = prop_mot_dec + parallax_dec + planetary_dec
    
    return(signal_ra, signal_dec)

# TESTING 
def generate_planet_signal_test(alpha, delta, A, F, B, G, a_AU, parallax, P_orb, t_peri, e, a_hat, times):
    sind, cosd, sina, cosa  = sin(delta), cos(delta), sin(alpha), cos(alpha)

    C = np.sqrt(a_hat**2-A**2-B**2)
    H = -np.sqrt(a_hat**2-F**2-G**2)

    M = (2*np.pi)*(times - t_peri)/P_orb  
    E = np.vectorize(E_from_M)(e, M)

    X = (cos(E)-e)   
    Y = np.sqrt((1-e**2))*sin(E)

    DELTA_X = A*X+F*Y             
    DELTA_Y = B*X+G*Y             
    DELTA_Z = H*X+C*Y   

    planetary_ra  = (1/cosd) * (sina*DELTA_X-cosa*DELTA_Y) 
    planetary_dec = (-cosd*DELTA_Z + sind*(DELTA_X*cosa+DELTA_Y*sina))

    return (planetary_ra, planetary_dec)


def generate_astrometry_signal_TEST(alpha, delta, parallax, mu_alpha, mu_delta, A, F, B, G, e, a_AU, P_orb, t_peri, times, a_earth = 1):

    a_hat = 0.014116666278885887*a_AU*parallax
    
    prop_mot_ra, prop_mot_dec = generate_pm_signal(alpha, delta, mu_alpha, mu_delta, times)
    
    parallax_ra, parallax_dec = generate_parallax_signal(alpha, delta, parallax, times)
    
    planetary_ra, planetary_dec = generate_planet_signal_test(alpha, delta, A, F, B, G, a_AU, parallax, P_orb, t_peri, e, a_hat, times)
    
    signal_ra  = prop_mot_ra  + parallax_ra  + planetary_ra
    signal_dec = prop_mot_dec + parallax_dec + planetary_dec
    
    return(signal_ra, signal_dec)

