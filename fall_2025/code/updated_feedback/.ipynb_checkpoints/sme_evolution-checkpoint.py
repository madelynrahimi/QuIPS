#Create master SME evolution doc for all types of feedback 

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.optimize import curve_fit

@njit(cache=True)
#Make effective temperature versus splitting plots 
def sme_evolution(del_t,period_number, eta, k, gamma, x0, p0, feedback, filt=None):
    w_real = 50000 * 2*np.pi  #rad/s
    m_bead = 4.8e-18 #kg
    hbar = 1.054e-34 #Js

    w_scale = w_real
    m_scale = m_bead
    x_scale = np.sqrt(hbar/(m_scale*w_scale))
    p_scale = np.sqrt(hbar*m_scale*w_scale)
    t_scale = 1/w_scale
    
   
    w = w_real/w_scale
    m = m_bead/m_scale
    
    T = (2*np.pi)/w
    t = period_number*T
    steps = int(t/del_t)

    #Use arrays, instead of appending lists to speed up iteration
    t_arr = np.empty(steps)
    t_per_arr = np.empty(steps)
    exp_x = np.empty(steps)
    exp_p = np.empty(steps)
    var_x_arr = np.empty(steps)
    var_p_arr = np.empty(steps)
    cov_xp_arr = np.empty(steps)
    exp_x_sq = np.empty(steps)
    exp_p_sq = np.empty(steps)
    alpha_x_stream = np.empty(steps)
    alpha_p_stream = np.empty(steps)
    w_sq_arr = np.empty(steps)

    #initial conditions
    x = x0
    p = p0
    alpha_x = x 
    alpha_p = 0.0
    var_x = 1/(2*m*w)
    var_p = m*w/2
    cov_xp = 0.0
    x_sq = var_x + x**2
    p_sq = var_p + p**2

    epsilon = -2*gamma/w
    w_sq = w**2*(1-2*epsilon)
    phi = np.arctan2(p, x)
    phi_p = np.pi/2 - 2*phi
    x_rot = x
    p_rot = p
    alpha_x_rot = x
    alpha_p_rot = p 
    phi_rot = np.arctan2(p_rot,x_rot)
    phi_p_rot = np.pi/2 - 2*phi_rot
    
    exp_x[0] = x
    exp_p[0] = p
    var_x_arr[0] = var_x
    var_p_arr[0] = var_p 
    cov_xp_arr[0]= cov_xp

    exp_x_sq[0] = x_sq
    exp_p_sq[0] = p_sq
    t_arr[0] = 0.0
    t_per_arr[0] = 0.0
    
    alpha_x_stream[0] = alpha_x
    alpha_p_stream[0] = alpha_p

    #Force feedback and unfiltered measurement stream estimation 
    if feedback == 'force':
        for i in range(steps-1):
            del_W = np.random.normal(loc=0.0, scale=np.sqrt(del_t), size=None)
            del_Q = 4*eta*k*x*del_t + np.sqrt(2*eta*k)*del_W
        
        
            if not filt:
                x += (1/m*p*del_t) + (np.sqrt(8*eta*k)*var_x*del_W) 
                p += (-m* w**2*x*del_t) + (np.sqrt(8*eta*k)*cov_xp*del_W) -gamma*alpha_p*del_t
            else:
                x += (1/m*p*del_t) -8*eta*k*x*var_x*del_t + 2*var_x*del_Q 
                p += (-m* w**2*x*del_t) -8*eta*k*x*cov_xp*del_t + 2*cov_xp*del_Q - gamma*p*del_t
        
                
            var_x += ((2/m*cov_xp) - (8*eta*k*var_x**2)) * del_t
            var_p += ((-2*m*w**2*cov_xp) + 2*k - (8*eta*k*cov_xp**2)) * del_t
            cov_xp += (1/m*var_p - m*w**2*var_x - (8*eta*k*var_x*cov_xp)) *del_t
            
            x_sq = var_x + x**2
            p_sq = var_p + p**2 
            
            alpha_x = x + (del_W/(np.sqrt(8*eta*k*del_t)))
            alpha_x_stream[i+1] = alpha_x
    
            t_now = (i+1) * del_t
            t_arr[i+1] = t_now
            t_per_arr[i+1] = t_now/T
            exp_x[i+1] = x
            exp_p[i+1] = p
            var_x_arr[i+1] = var_x
            var_p_arr[i+1] = var_p
            cov_xp_arr[i+1] = cov_xp
            exp_x_sq[i+1] = x_sq
            exp_p_sq[i+1] = p_sq
            
            if i >= 1:
                alpha_p = m*(alpha_x_stream[i+1] - alpha_x_stream[i])/del_t
            else:
                alpha_p = 0.0
            alpha_p_stream[i+1] = alpha_p
            
    else:
        for i in range(steps-1):
            del_W = np.random.normal(0.0, np.sqrt(del_t))
            del_Q = 4*eta*k*x*del_t + np.sqrt(2*eta*k)*del_W
            if not filt:
                x += (1/m*p*del_t) + (np.sqrt(8*eta*k)*var_x*del_W) 
                p += (-m* w_sq*x*del_t) + (np.sqrt(8*eta*k)*cov_xp*del_W) 
            else:
                x += (1/m*p*del_t) -8*eta*k*x*var_x*del_t + 2*var_x*del_Q 
                p += (-m* w_sq*x*del_t) -8*eta*k*x*cov_xp*del_t + 2*cov_xp*del_Q
                
        
                
            var_x += ((2/m*cov_xp) - (8*eta*k*var_x**2)) * del_t
            var_p += ((-2*m*w_sq*cov_xp) + 2*k - (8*eta*k*cov_xp**2)) * del_t
            cov_xp += (1/m*var_p - m*w_sq*var_x - (8*eta*k*var_x*cov_xp)) *del_t
            
            x_sq = var_x + x**2
            p_sq = var_p + p**2
    
            alpha_x = x + (del_W/(np.sqrt(8*eta*k*del_t)))
            alpha_x_stream[i+1] = alpha_x
            
            t_now = (i+1) * del_t
            t_arr[i+1] = t_now
            t_per_arr[i+1] = t_now/T
            exp_x[i+1] = x
            exp_p[i+1] = p
            var_x_arr[i+1] = var_x
            var_p_arr[i+1] = var_p
            cov_xp_arr[i+1] = cov_xp
            exp_x_sq[i+1] = x_sq
            exp_p_sq[i+1] = p_sq
            
            if i >= 1:
                alpha_p = m*(alpha_x_stream[i+1] - alpha_x_stream[i])/del_t
            else:
                alpha_p = 0.0
            alpha_p_stream[i+1] = alpha_p

                
            
    
            #Redefine, x_rot and p_rot using measurement stream values of x and p 
            if not filt:
                x_fb = alpha_x
                p_fb = alpha_p
            else:
                x_fb = x
                p_fb = p
                
            x_rot = x_fb*np.cos(w*t_arr[i+1]) - p_fb*np.sin(w*t_arr[i+1])
            p_rot = x_fb*np.sin(w*t_arr[i+1]) + p_fb*np.cos(w*t_arr[i+1])
            
            
            phi_rot = np.arctan2(p_rot, x_rot)
            phi_p_rot = np.pi/2 - 2*phi_rot

            w_sq = w**2*(1 - 2*epsilon*np.cos(2*w*t_arr[i+1] + phi_p_rot))
            w_sq_arr[i+1] = w_sq
        
            
            
            
        
        
    t_real_arr = t_arr *t_scale
    exp_x_real  = exp_x * x_scale
    exp_p_real  = exp_p * p_scale
    var_x_real = var_x_arr * x_scale**2
    var_p_real = var_p_arr * p_scale**2
    cov_xp_real = cov_xp_arr * x_scale*p_scale
    exp_x_sq_real = exp_x_sq * x_scale**2
    exp_p_sq_real = exp_p_sq * p_scale**2

    
    
    alpha_x_stream_real = alpha_x_stream * x_scale
    alpha_p_stream_real = alpha_p_stream * p_scale
    
    
    return (t_real_arr, t_per_arr,exp_x_real,exp_p_real, alpha_x_stream_real, alpha_p_stream_real, var_x_real, var_p_real, cov_xp_real, exp_x_sq_real,exp_p_sq_real,w_sq_arr)

def plot_sme_traj(del_t,period_number, eta, k, gamma, x0, p0, feedback, filt=None):
    t_real_arr, t_per_arr,exp_x_real,exp_p_real, alpha_x_stream_real, alpha_p_stream_real, var_x_real, var_p_real, cov_xp_real,exp_x_sq_real,exp_p_sq_real,w_sq_arr = sme_evolution(del_t=del_t, period_number=period_number, eta=eta, k=k, gamma=gamma, x0=x0, p0=p0, feedback=feedback, filt=filt)
    if not filt:
        filt_tag = "Measurement Stream"
    elif filt == "kalman":
        filt_tag = "Kalman Filter"
    if feedback == "force":
        feed_tag = "Force Feedback"
    elif feedback == "parametric":
        feed_tag = "Parametric Feedback"
    plt.figure()
    plt.plot(t_real_arr*10**6, exp_x_real)
    plt.xlabel("t ($\mu s$)")
    plt.ylabel("<x> (m)")
    plt.title(f"{filt_tag} {feed_tag} Position Expectation Value Time Evolution")
    plt.show()
    
    plt.figure()
    plt.plot(t_real_arr*10**6, exp_p_real)
    plt.xlabel("t ($\mu s$)")
    plt.ylabel("<p> ($kg \cdot m \cdot s^{-1}$)")
    plt.title(f"{filt_tag} {feed_tag} Momentum Expectation Value Time Evolution")
    plt.show()
    
    plt.figure()
    plt.plot(t_real_arr*10**6, alpha_x_stream_real)
    plt.xlabel("t ($\mu s$)")
    plt.ylabel(r"$\alpha_{x}$ (m)")
    plt.title(f"{filt_tag} {feed_tag} Experiment Position Stream Time Evolution")
    plt.show()

def t_eff(p_sq, x_sq, m, w):
    kB = 1.3806*10**-23
    
    T_eff = (p_sq/m + m*w**2*x_sq)/(2*kB)
    return T_eff
def occ_num(T_list):
    hbar = 1.054e-34
    kB = 1.381e-23

    omega = 50000*2*np.pi   #rad/s

    n_arr = 1/(np.exp(hbar*omega/(kB*np.array(T_list))) - 1)
    return n_arr
feedback_style_dict = {"force": {"color": "tab:blue","label": "Force Feedback"},"parametric": {"color": "tab:orange","label": "Parametric Feedback"}}

filter_style_dict = {None: {"linestyle": "-","label": "Measurement Stream"},"kalman": {"linestyle": "--","label": "Kalman Filter"}}

def effective_temp_plot(del_t,period_number, eta, k, gamma_list, x0, p0, feedback, temp_bool = True, filt=None, m_bead=4.8e-18, w_real=50000*2*np.pi, power_tag = True, plot_bool = False):
    T_list = []
    P_list = []
    k_list = []
    for gamma in gamma_list:
        t_real_arr, t_per_arr,exp_x_real,exp_p_real, alpha_x_stream_real, alpha_p_stream_real, var_x_real, var_p_real, cov_xp_real,exp_x_sq_real,exp_p_sq_real,w_sq_arr = sme_evolution(del_t=del_t, period_number=period_number, eta=eta, k=k, gamma=gamma, x0=x0, p0=p0, feedback=feedback, filt=filt)
        steps = len(t_real_arr)
        average_increment = int(steps/4)
        x_sq_real = np.mean(exp_x_sq_real[-average_increment:])
        p_sq_real = np.mean(exp_p_sq_real[-average_increment:])
    
        t_e = t_eff(p_sq_real, x_sq_real, m=m_bead, w=w_real)
        T_list.append(t_e)
        if power_tag:
            _, I, P = las_power_traj(del_t,period_number, eta, k, gamma, x0, p0, feedback='parametric',w_sq_prev = w_sq_arr, filt=None, run_bool = False, verbose=False)
            k_t= meas_strength_traj(del_t,period_number, eta, k, gamma, x0, p0, feedback='parametric', w_sq_prev = w_sq_arr, filt=None,run_bool = False, verbose=False)
            P_max = max(P)
            k_max = max(k_t)
            
            P_list.append(P_max)
            k_list.append(k_max)
    if temp_bool:
        state_arr = np.array(T_list)
        state_title = "Effective Temperature"
        state_tag = "T (K)"
    else:
        state_arr = occ_num(T_list)
        state_title = "Occupation Number"
        state_tag = "n"
    gamma_list_real = gamma_list* (w_real/(2*np.pi))
        
    
    feedback_style = feedback_style_dict[feedback]
    filter_style = filter_style_dict[filt]
    label = (f"{feedback_style['label']}, "f"{filter_style['label']}, "+ rf"$\eta={eta}$, $k={k}$")    

    if plot_bool and power_tag is False:
        fig, ax = plt.subplots()
        ax.plot(gamma_list_real,state_arr,color=feedback_style["color"],linestyle=filter_style["linestyle"],marker="o",label=label)
    
        ax.set_xlabel(r"$\Gamma$ (Hz)")
        ax.set_ylabel(f"{state_tag}")
        ax.set_title(f"{state_title} vs. Damping Rate")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        plt.show()
    elif plot_bool and power_tag:
        fig, ax1 = plt.subplots()
        ax1.plot(state_arr, np.array(P_list), marker="o", label=r"$P_{\max}$")
        if temp_bool is False:
            ax1.axvline(x=1, color='black', linestyle='--')
        ax1.set_xlabel(f"{state_tag}")
        ax1.set_ylabel(r"$P_{\max}$ (W)")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        
        # Right y-axis: gamma
        ax2 = ax1.twinx()
        ax2.plot(state_arr, gamma_list_real, marker="s", linestyle="--", label=r"$\Gamma$")
        ax2.set_ylabel(r"$\Gamma$ (Hz)")
        ax2.set_yscale("log")
        
        ax1.set_title(f"Max Power and Damping Rate vs. {state_title}")
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)
        plt.show()
    
    return [T_list, P_list, k_list]


def comp_effective_temp_plot(del_t,period_number, eta_list, k_list, gamma_list, x0, p0, feedback_list, filt_list, m_bead=4.8e-18, w_real=50000*2*np.pi):
    T_comp_list = []
    gamma_list_real = gamma_list* (w_real/(2*np.pi))
    fig, ax = plt.subplots()
    for eta, k, feedback, filt in zip(eta_list, k_list, feedback_list, filt_list):
        T_list = (effective_temp_plot(del_t,period_number, eta, k, gamma_list, x0, p0, feedback, filt))
        T_comp_list.append(T_list)
        feedback_style = feedback_style_dict[feedback]
        filter_style = filter_style_dict[filt]
        label = (f"{feedback_style['label']}, "f"{filter_style['label']}, "+ rf"$\eta={eta}$, $k={k}$")
        ax.plot(gamma_list_real,np.array(T_list),color=feedback_style["color"],linestyle=filter_style["linestyle"],marker="o",label=label)
    ax.set_xlabel(r"$\Gamma$ (Hz)")
    ax.set_ylabel("T (K)")
    ax.set_title("Effective Temperature vs. Damping Rate")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    plt.show()
    return T_comp_list

def ensemble_av_traj(del_t,period_number, eta, k, gamma, x0, p0, feedback, traj_num, filt=None):
    x_sq_traj = []
    p_sq_traj = []
    w_real = 50000 * 2*np.pi  #rad/s
    m_bead = 4.8e-18 #kg
    
    for i in range(traj_num):
        rng = np.random.default_rng()
        initial_arr = rng.uniform(low=-300.0, high=300.0, size=2)
        x0 = initial_arr[0]
        p0= initial_arr[1]
        t_real_arr, t_per_arr,exp_x_real,exp_p_real, alpha_x_stream_real, alpha_p_stream_real, var_x_real, var_p_real, cov_xp_real,exp_x_sq_real,exp_p_sq_real,w_sq_arr = sme_evolution(del_t=del_t, period_number=period_number, eta=eta, k=k, gamma=gamma, x0=x0, p0=p0, feedback=feedback, filt=filt)
        x_sq_traj.append(exp_x_sq_real)
        p_sq_traj.append(exp_p_sq_real)
    x_sq_ensemble = np.mean(x_sq_traj, axis=0)
    p_sq_ensemble = np.mean(p_sq_traj, axis=0)
    temp_traj = t_eff(p_sq_ensemble, p_sq_ensemble, m=m_bead, w=w_real)
    return t_real_arr, temp_traj

def exp_decay(t, T_eff, A, alpha):
    return T_eff + (A*np.exp(-alpha*t))

def cooling_rate_calc(del_t,period_number, eta, k, gamma, x0, p0, n_bins,feedback, traj_num, filt=None, plot_bool= False):
    t_real_arr, temp_traj = ensemble_av_traj(del_t=del_t,period_number=period_number, eta=eta, k=k, gamma=gamma, x0=x0, p0=p0, feedback=feedback, traj_num=traj_num, filt=filt)
    bins = np.linspace(t_real_arr.min(), t_real_arr.max(), n_bins+1)
    t_top_env = []
    T_top_env = []
    for i in range(n_bins):
        mask = (t_real_arr>=bins[i]) & (t_real_arr<bins[i+1])
        t_bin = t_real_arr[mask]
        T_bin = temp_traj[mask]
        max_idx = np.argmax(T_bin)
        t_top_env.append(t_bin[max_idx])
        T_top_env.append(T_bin[max_idx])
    p0 = [T_top_env[-1], T_top_env[0] - T_top_env[-1], 1e3]
    params, cov = curve_fit(exp_decay, t_top_env,T_top_env, p0=p0)
    T_eff_fit, A_fit, alpha_fit = params
    t_fit_arr = np.linspace(0,t_top_env[-1], 10000)
    T_fit_arr = exp_decay(t_fit_arr, T_eff_fit, A_fit, alpha_fit)

    if not filt:
        filt_tag = "Measurement Stream"
    elif filt == "kalman":
        filt_tag = "Kalman Filter"
    if feedback == "force":
        feed_tag = "Force Feedback"
    elif feedback == "parametric":
        feed_tag = "Parametric Feedback"
    fit_text = (
    r"$T(t) = T_{\mathrm{eff}} + A e^{-\gamma t}$" "\n"
    rf"$\eta = {eta}$" "\n"
    rf"$k = {k}$" "\n"
    rf"$\Gamma = {gamma}$" "\n"
    rf"$T_{{\mathrm{{eff}}}} = {T_eff_fit*1e6:.2f}\ \mu\mathrm{{K}}$" "\n"
    rf"$A = {A_fit*1e6:.2f}\ \mu\mathrm{{K}}$" "\n"
    rf"$\gamma = {alpha_fit:.2e}\ \mathrm{{s^{{-1}}}}$")



    if plot_bool:
        plt.figure()
        plt.scatter(np.array(t_top_env)*10**6, np.array(T_top_env)*10**6, s=8)
        plt.plot(t_fit_arr*10**6, T_fit_arr*10**6)
        plt.xlabel("t ($\mu s$)")
        plt.ylabel("Temperature ($\mu K$)")
        plt.text(0.98, 0.98, fit_text,transform=plt.gca().transAxes,ha='right', va='top',fontsize=10)
        plt.title(f"{filt_tag} {feed_tag} Temperature vs. Time of Feedback Application")
        plt.show()
    print(f"Time to feedback cool {1e6/alpha_fit:0.2f} microseconds")
    return params
        
    
def feedback_time_plot(del_t,period_number, eta, k, gamma, x0, p0, feedback, traj_num, filt=None):
    t_real_arr, temp_traj = ensemble_av_traj(del_t=del_t,period_number=period_number, eta=eta, k=k, gamma=gamma, x0=x0, p0=p0, feedback=feedback, traj_num=traj_num, filt=filt)
    if not filt:
        filt_tag = "Measurement Stream"
    elif filt == "kalman":
        filt_tag = "Kalman Filter"
    if feedback == "force":
        feed_tag = "Force Feedback"
    elif feedback == "parametric":
        feed_tag = "Parametric Feedback"
    plt.figure()
    plt.scatter(t_real_arr*10**6, temp_traj*10**6)
    plt.xlabel("t ($\mu s$)")
    plt.ylabel("Temperature ($\mu K$)")
    plt.text(0.95, 0.85, f"$\\Gamma = {gamma}$",transform=plt.gca().transAxes, ha='right', va='top')
    plt.text(0.95, 0.95, f"$\\eta = {eta}$",transform=plt.gca().transAxes, ha='right', va='top')
    plt.text(0.95, 0.90, f"$k = {k}$",transform=plt.gca().transAxes, ha='right', va='top')
    plt.title(f"{filt_tag} {feed_tag} Temperature vs. Time of Feedback Application")
    
    plt.show()

def k_0(P_0=0.5, NA=0.77, f=0.0031):
    w_real = 50000 * 2*np.pi  #rad/s
    m_bead = 4.8e-18 #kg
    hbar = 1.054e-34 #Js
    c = 3*10**8
    
    w_scale = w_real
    m_scale = m_bead
    
    x_scale = np.sqrt(hbar/(m_scale*w_scale))
    
    
    A = 0.831670643272 #Calculated using NA 0.77
    laser_wavelength = 1064e-9 #m
    k = (2*np.pi)/laser_wavelength
    q_sq = k**2*((2/5)+A**2) #m^-2
    
    #Using epsilon for 1064nm wavelength laser and 100 nm diameter bead
    epsilon = 2.10
    bead_rad = 50e-9 #m 
    V = (4/3)*np.pi*bead_rad**3

    alpha_1 = 3*V*(epsilon - 1)/(epsilon + 2)
    alpha_2 = (k**3 *alpha_1**2)/(6*np.pi) #m^3

    #I_0 = P_0/(np.pi*f**2*NA**2)
    I_0 = 5*10**10
    k_unit = I_0*alpha_2*q_sq/(4*hbar*c) #m^-2*s^-1
    k = k_unit*(x_scale**2)/w_scale

    return k
def las_power_traj(del_t,period_number, eta,k, gamma, x0, p0, feedback='parametric',w_sq_prev = None, filt=None, run_bool = True, verbose=True):
    P_0 = 0.5 #W
    NA = 0.77 #unitless
    f = 0.0031 # m
    w = 1 #w_sq arr in unitless dimensions
    if run_bool:
        t_real_arr, t_per_arr,exp_x_real,exp_p_real, alpha_x_stream_real, alpha_p_stream_real, var_x_real, var_p_real, cov_xp_real,exp_x_sq_real,exp_p_sq_real,w_sq_arr = sme_evolution(del_t=del_t, period_number=period_number, eta=eta, k=k, gamma=gamma, x0=x0, p0=p0, feedback=feedback, filt=filt)
    else:
        w_sq_arr = w_sq_prev
        t_real_arr = []
        
    I = (P_0/(np.pi*f**2*NA**2))*(w_sq_arr/w**2)
    P = P_0*(w_sq_arr/w**2)
    if verbose:
        plt.figure()
        plt.plot(t_real_arr[1:]*10**6, I[1:])
        plt.xlabel("t ($\mu s$)")
        plt.ylabel(r"$I\ \left(\frac{W}{m^2}\right)$")
        plt.title(f"Laser Intensity vs. Time with $\Gamma$ = {gamma}")
        plt.show()
        
        plt.figure()
        plt.plot(t_real_arr[1:]*10**6, P[1:])
        plt.xlabel("t ($\mu s$)")
        plt.ylabel("P (W)")
        plt.title(f"Laser Power vs. Time with $\Gamma$ = {gamma}")
        plt.show()
        
    return [t_real_arr,I,P]
    
def meas_strength_traj(del_t,period_number, eta, k, gamma, x0, p0, w_sq_prev, feedback='parametric',filt=None,run_bool = False, verbose=True):
    
    w_real = 50000 * 2*np.pi  #rad/s
    m_bead = 4.8e-18 #kg
    hbar = 1.054e-34 #Js

    w_scale = w_real
    m_scale = m_bead
    
    x_scale = np.sqrt(hbar/(m_scale*w_scale))
    
    NA = 0.77
    
    t_real_arr, I, P = las_power_traj(del_t,period_number, eta, k, gamma, x0, p0, w_sq_prev = w_sq_prev, feedback='parametric', filt=None, run_bool=run_bool, verbose = False)

    
    A = 0.831670643272 #Calculated using NA 0.77
    laser_wavelength = 1064/10e9 #m
    k = (2*np.pi)/laser_wavelength
    q_sq = k**2*((2/5)+A**2) #m^-2
    
    #Using epsilon for 1064nm wavelength laser and 100 nm diameter bead
    epsilon = 2.10
    bead_rad = 50/10e9 #m 
    V = (4/3)*np.pi*bead_rad**3

    alpha_1 = 3*V*(epsilon - 1)/(epsilon + 2)
    alpha_2 = (k**3 *alpha_1**2)/(6*np.pi) #m^3

    k_t_unit = I*alpha_2*q_sq/(4*hbar) #m^-2*s^-1
    k_t = k_t_unit * x_scale**2
    if verbose:
        plt.figure()
        plt.plot(t_real_arr[1:]*10**6, k_t[1:])
        plt.xlabel("t ($\mu s$)")
        plt.ylabel(r"$k\ \left(\frac{1}{m^2\,s^1}\right)$")
        plt.title(f"Measurement Strength vs. Time with $\Gamma$ = {gamma}")
        plt.show()
    return k_t




    

    
    
    
    


    
    

#left and right axes of phonon number vs temperature 
            