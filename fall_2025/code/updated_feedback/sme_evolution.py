#Create master SME evolution doc for all types of feedback 

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

#@njit(cache=True)
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
    
    
    return (t_real_arr, t_per_arr,exp_x_real,exp_p_real, alpha_x_stream_real, alpha_p_stream_real, var_x_real, var_p_real, cov_xp_real, exp_x_sq_real,exp_p_sq_real)

def plot_sme_traj(del_t,period_number, eta, k, gamma, x0, p0, feedback, filt=None):
    t_real_arr, t_per_arr,exp_x_real,exp_p_real, alpha_x_stream_real, alpha_p_stream_real, var_x_real, var_p_real, cov_xp_real,exp_x_sq_real,exp_p_sq_real = sme_evolution(del_t=del_t, period_number=period_number, eta=eta, k=k, gamma=gamma, x0=x0, p0=p0, feedback=feedback, filt=filt)
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

feedback_style_dict = {"force": {"color": "tab:blue","label": "Force Feedback"},"parametric": {"color": "tab:orange","label": "Parametric Feedback"}}

filter_style_dict = {None: {"linestyle": "-","label": "Measurement Stream"},"kalman": {"linestyle": "--","label": "Kalman Filter"}}

def effective_temp_plot(del_t,period_number, eta, k, gamma_list, x0, p0, feedback, filt=None, m_bead=4.8e-18, w_real=50000*2*np.pi, plot_bool = False):
    T_list = []
    for gamma in gamma_list:
        t_real_arr, t_per_arr,exp_x_real,exp_p_real, alpha_x_stream_real, alpha_p_stream_real, var_x_real, var_p_real, cov_xp_real,exp_x_sq_real,exp_p_sq_real = sme_evolution(del_t=del_t, period_number=period_number, eta=eta, k=k, gamma=gamma, x0=x0, p0=p0, feedback=feedback, filt=filt)
        steps = len(t_real_arr)
        average_increment = int(steps/4)
        x_sq_real = np.mean(exp_x_sq_real[-average_increment:])
        p_sq_real = np.mean(exp_p_sq_real[-average_increment:])
    
        t_e = t_eff(p_sq_real, x_sq_real, m=m_bead, w=w_real)
        T_list.append(t_e)
    gamma_list_real = gamma_list* (w_real/(2*np.pi))
    
    feedback_style = feedback_style_dict[feedback]
    filter_style = filter_style_dict[filt]
    label = (f"{feedback_style['label']}, "f"{filter_style['label']}, "+ rf"$\eta={eta}$, $k={k}$")    

    if plot_bool:
        fig, ax = plt.subplots()
        ax.plot(gamma_list_real,np.array(T_list),color=feedback_style["color"],linestyle=filter_style["linestyle"],marker="o",label=label)
    
        ax.set_xlabel(r"$\Gamma$ (Hz)")
        ax.set_ylabel("T (K)")
        ax.set_title("Effective Temperature vs. Damping Rate")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        plt.show()
    
    return T_list


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
        