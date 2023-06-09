import numpy as np 
from numpy import cos, sin, arcsin, arccos, arctan
import rebound 
AU_pc   = 4.84814e-6


def get_sim(d, delta, alpha, m_star, vx_star, vy_star, vz_star, m_planet, a_AU, e,omega,Omega, inc):
    # with planet 
	x_star = d*cos(alpha)*cos(delta)
	y_star = d*cos(delta)*sin(alpha)
	z_star = d*sin(delta)

	sim_wp = rebound.Simulation()                            # create a simulation named sim_wp 
	sim_wp.units = ["msun","AU","year"]                      # setting units of sim_wpm

	sim_wp.add(m = 1)                                        # add the Sun as the central object 

	sim_wp.add(m = 3.0027e-6, a = 1, e = 0)                  # add Earth in a 1 AU circular orbit 
	# add a star 50 pc away with calculated velocity and set parameters

	sim_wp.add(x = x_star,y=y_star, z=z_star, vx=vx_star,vy = vy_star,m = m_star,vz = vz_star)   
	sim_wp.add(m = m_planet, a = a_AU, e = e, primary = sim_wp.particles[2], inc = inc, M=0,omega=omega,Omega=Omega)  # add planet from figure 3.1 caption and have it orbit the star 

	# barycentre particles
	com_particle  = sim_wp.calculate_com(first = 2,last = 4)
	ssbc_particle = sim_wp.calculate_com(first = 0,last = 2)

	# without planet
	sim = rebound.Simulation()                              # create a simulation named sim 

	sim.units = ["msun","AU","year"]                        # setting units of sim 

	sim.add(m = 1)                                          # add the Sun as the central object 

	sim.add(m = 3.0027e-6, a = 1, e = 0)                    # add Earth in a 1 AU circular orbit 

	sim.add(com_particle)                                   # add a particle equivaent to the star-planet barycentre

	return(sim_wp, sim)


def sim_signal(sim, times, m):
	# making arrays filled with zeros 
	r_ssbc, r_earth, r_star, r_spbc = np.zeros((m,3)), np.zeros((m,3)), np.zeros((m,3)), np.zeros((m,3))
	v_ssbc, v_spbc,  = np.zeros((m,3)), np.zeros((m,3))
	R_SE, delta_SE, alpha_SE = np.zeros((m,3)), np.zeros(m), np.zeros(m)
	R_SC, delta_SC, alpha_SC = np.zeros((m,3)), np.zeros(m), np.zeros(m)

	for i,t in enumerate(times):
		# integrate simulations
		sim.integrate(t)
		
		# set particles 
		ssbc_particle   = sim.calculate_com(first = 0,last = 2)  # solar system barycentre 
		earth_particle  = sim.particles[1]                       # earth 
		star_particle   = sim.particles[2]                       # star 
		spbc_particle   = sim.calculate_com(first = 2,last = 4)  # star-planet barycentre
		
		# positions of particles 
		r_ssbc[i] = np.array(ssbc_particle.xyz)
		r_earth[i] = np.array(earth_particle.xyz)
		r_star[i] = np.array(star_particle.xyz)
		r_spbc[i] = np.array(spbc_particle.xyz)
		
		# position vectors 
		R_SE[i] = r_star[i] - r_earth[i]  # spbc - earth 
		R_SC[i] = r_spbc[i] - r_ssbc[i]   # spbc - ssbc 

		# using simple_function to turn position vectros into delta and alpha
		_, delta_SE[i], alpha_SE[i] = simple_func(R_SE[i])  # star - earth 
		_, delta_SC[i], alpha_SC[i] = simple_func(R_SC[i])  # star - ssbc 
		
		# velocities 
		v_ssbc[i]   = np.array(ssbc_particle.vxyz)    # ssbc
		v_spbc[i]   = np.array(spbc_particle.vxyz)    # star  
		
		# initial conditions needed for other functions 
		R_SC_0 = r_spbc[0] - r_ssbc[0]
		V_SC_0 = v_spbc[0] - v_ssbc[0]
			
	return(delta_SE, alpha_SE, delta_SC, alpha_SC, R_SC_0, V_SC_0)


def simple_func(R):
	Rx, Ry, Rz = R

	rho = np.sqrt(Rx**2+Ry**2)

	d      = np.linalg.norm(R, axis=0)
	delta  = np.arctan2(Rz, rho)
	alpha  = np.arctan2(Ry, Rx)

	return(d, delta, alpha)

def E_from_M(e,M, tol=1e-10):
	E = M
	f=np.inf
	while np.abs(f) > (tol):
		f = E-e*sin(E)-M
		f_prime = 1-e*cos(E)
		E = E -f/f_prime
	return(E)


def proper_motion_eq(R, V, times, t0):
	# finding initial values of d_bc, delta_bc and alpha_bc 
	d, delta, alpha = simple_func(R)

	# finding initial values of d_dot_bc, delta_dot_bc and alpha_dot_bc
	Vx, Vy, Vz = V 
	sind, cosd, sina, cosa  = sin(delta), cos(delta), sin(alpha), cos(alpha)

	d_dot     =   Vx * cosa * cosd      +  Vy * sina * cosd     +  Vz * sind
	delta_dot = -(Vx * sind * cosa)/d   - (Vy * sina * sind)/d  + (Vz * cosd)/d  # proper motion in dec direction [rad/year]
	alpha_dot = -(Vx * sina)/(d * cosd) + (Vy * cosa)/(d*cosd)                   # proper motion in ra direction  [rad/year]

	T = times - t0  # time variable T [years]

	pm_term_dec = delta_dot*T  # proper motion term in dec direction [rad]
	pm_term_ra  = alpha_dot*T  # proper motion term in ra direction  [rad]

	return (pm_term_dec, pm_term_ra)


def parallax_eq(R, times, t0, a_earth):
	d, delta, alpha = simple_func(R)
	sind, cosd, sina, cosa  = sin(delta), cos(delta), sin(alpha), cos(alpha)

	T = times - t0  # time variable T [years]

	parallax_dec = (a_earth/d)*sind*cos(2*np.pi*T-alpha)                                     #    [rad]
	parallax_ra  = (a_earth/(d*cosd))*(sina*cos(2*np.pi*T)-cosa*sin(2*np.pi*T))              #    [rad]
      
	return(parallax_dec, parallax_ra)
 
def thiele_innes(R, m_star, m_planet, omega, Omega, inc):
	d, delta, alpha = simple_func(R)

	a_hat = (m_planet/(m_star+m_planet)) * 1/d

	A = a_hat * (  cos(omega)  * cos(Omega) - sin(omega) * sin(Omega) * cos(inc))  #                      [as]
	F = a_hat * ( - sin(omega) * cos(Omega) - cos(omega) * sin(Omega) * cos(inc))  #                      [as]

	B = a_hat * (  cos(omega)  * sin(Omega) + sin(omega) * cos(Omega) * cos(inc))  #                      [as]
	G = a_hat * ( - sin(omega) * sin(Omega) + cos(omega) * cos(Omega) * cos(inc))  #                      [as]

	H = a_hat * sin(omega)*sin(inc)
	C = a_hat * sin(inc)*cos(omega)

	return(A, F, B, G, H, C)

def planetary_eq(R, m_star, m_planet, omega, Omega, inc, e, a_AU, times, tau, G_const=4*np.pi**2):
	d, delta, alpha = simple_func(R)
	sind, cosd, sina, cosa  = sin(delta), cos(delta), sin(alpha), cos(alpha)

	A, F, B, G, H, C = thiele_innes(R, m_star, m_planet, omega, Omega, inc)

	# planetary term
	rtGM = np.sqrt(G_const * (m_star+m_planet))
	P = 2*np.pi * a_AU**(3/2)/rtGM  # period of the planet              [years]


	M = (2*np.pi)*(times - tau)/P  # mean anomaly [rad]
	E = np.vectorize(E_from_M)(e,M)

	X = a_AU * (cos(E)-e)   
	Y = a_AU * (np.sqrt((1-e**2))*sin(E))


	DELTA_X = A*X+F*Y             
	DELTA_Y = B*X+G*Y             
	DELTA_Z = H*X+C*Y   

	planetary_dec = (-cosd*DELTA_Z + sind*(DELTA_X*cosa+DELTA_Y*sina))
	planetary_ra  = (1/cosd) * (sina*DELTA_X-cosa*DELTA_Y) 

	return(planetary_dec, planetary_ra)





