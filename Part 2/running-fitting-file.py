# importing packages, functions, etc 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from error_var_fitting import big_func

# initial inputs 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
N = 100 # number of timesteps 
S = 100  # number of errors
K = 20   # number of periods
J = 20   # number of entries in each bin

times = np.linspace(0, 5, N)
theta = np.linspace(0, 2*np.pi, N)

alpha0, delta0 = 1, 0.3

colors = sns.color_palette("hls", J*K)
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


# plotting
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
for k in range(K):
    for j in range(J):
        # marker = markers[2*k+j]
        color = colors[2*k+j]
        plt.scatter(a_hat[j,k]/sigma_err[j,k], -1*Delta_BIC[j,k], label= P_orb_array[k], color=colors[2*k+j])


plt.xscale('log')
plt.yscale('log')


plt.xlabel('S/N', fontsize = 15)
plt.ylabel('$-\Delta$BIC', fontsize=15)
plt.title('-$\Delta$BIC vs S/N',fontsize=20)

plt.axhline(20, color='gray')
plt.text(5.9, 23, '-$\Delta$BIC = 20', fontsize = 15, color='gray')
plt.show
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
for k in range(K):
    for j in range(J):
        plt.scatter(a_hat[j,k]/sigma_err[j,k], detection_frac[k],  label= P_orb_array[k], color=colors[2*k+j])


# plt.xscale('log')
# plt.yscale('log')


plt.axhline(20, color='gray')
plt.text(5.9, 23, '-$\Delta$BIC = 20', fontsize = 15, color='gray')


# plot legend
# legend=plt.legend(loc="lower right",frameon=True, markerscale = 1, bbox_to_anchor=(1.2, 0.4), title="$P_{orb}$ Value")
# legend.get_frame().set_edgecolor('0.3')
# legend.get_frame().set_linewidth(1)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
for j in range(J):
        for k in range(K):
            plt.hexbin(SN[j,k], P_orb_plot_array[k], C = detection_frac[k], gridsize=5, cmap='YlGn')


cb = plt.colorbar(shrink=0.9, label="% det.")
plt.title('Figure 5', fontsize=20)
plt.xlabel('$S/N$', fontsize=15)
plt.ylabel('$P_{orb}$ [years]', fontsize=15)


plt.show
# ---------------------------------------------------------------------------------------------------------------------------------------------------------