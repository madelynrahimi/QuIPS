#Create master SME evolution doc for all types of feedback 

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import os

#@njit(cache=True)
def sme_evolution_2D(del_t,period_number, eta, k, gamma, x0, p0, feedback, fx = 5e4, fy = 2e4, filt=None, M=None):
    
    wx_real = fx * 2*np.pi  #rad/s
    wy_real = fy * 2*np.pi  #rad/s
    m_bead = 4.8e-18 #kg
    hbar = 1.054e-34 #Js

    w0 = wx_real
    m_scale = m_bead
    
    x_scale = np.sqrt(hbar / (m_scale * w0))
    p_scale = np.sqrt(hbar * m_scale * w0)
    t_scale = 1 / w0
    
    wx = wx_real / w0    # = 1
    wy = wy_real / w0    # = fy/fx
    
    x_scale_vec = np.array([x_scale, x_scale])
    p_scale_vec = np.array([p_scale, p_scale])
    
    Sx = x_scale * np.eye(2)
    Sp = p_scale * np.eye(2)

    w_arr = np.array([wx, wy]).reshape(2,1)
    w_arr_swapped = w_arr[::-1, :]
    

    m = m_bead/m_scale
    
    T = (2*np.pi)/wx
    t = period_number*T
    steps = int(t/del_t)
    K = np.zeros((2, 2))
    
    if M is None:
        M = m * np.eye(2)
    if feedback == 'force':
        K[0, 0] = m * wx**2
        K[1, 1] = m * wy**2
    else:
        Gamma = gamma * np.ones((2,1))
        epsilon = -2 * Gamma / w_arr
        epsilon_swapped = epsilon[::-1, :]
        w_sq = w_arr**2 * (1 - 2*epsilon)
        
        K = m * np.diag(w_sq.flatten())
        
    Gamma = gamma *np.eye(2)   
    M_inv = np.linalg.inv(M)
    
    

    #Use arrays, instead of appending lists to speed up iteration
    t_arr = np.empty(steps)
    t_per_arr = np.empty(steps)
    
    exp_x = np.empty((steps,2))
    exp_p = np.empty((steps,2))
    
    var_x_arr = np.empty((steps,2,2))
    var_p_arr = np.empty((steps,2,2))
    cov_xp_arr = np.empty((steps,2,2))
    
    exp_x_sq = np.empty((steps,2,2))
    exp_p_sq = np.empty((steps,2,2))

    alpha_x_stream = np.empty((steps,2))
    alpha_p_stream = np.empty((steps,2))


    #initial conditions
    x0 = np.array(x0, dtype=np.float64)
    p0 = np.array(p0, dtype=np.float64)
    x = x0.reshape(2,1)
    p = p0.reshape(2,1)
    alpha_x = x.copy() 
    alpha_p = np.zeros((2,1),dtype=float)
    
    Vxx = np.zeros((2,2),dtype=float)
    Vpp = np.zeros((2,2),dtype=float)
    Vxp = np.zeros((2,2),dtype=float)

    Vxx[0,0] = 1/(2*m*wx)
    Vxx[1,1] = 1/(2*m*wy)
    Vpp[0,0]= m*wx/2
    Vpp[1,1]= m*wy/2
    
    t_arr[0] = 0.0
    t_per_arr[0] = 0.0

    exp_x[0] = x[:,0]
    exp_p[0] = p[:,0]
    alpha_x_stream[0] = alpha_x[:,0]
    alpha_p_stream[0] = alpha_p[:,0]
    
    
    x_rot = x.copy()
    p_rot = p.copy()
  
    phi_rot = np.arctan2(p_rot,x_rot)
    phi_p_rot = np.pi/2 - 2*phi_rot
    
    
    var_x_arr[0] = Vxx
    var_p_arr[0] = Vpp 
    cov_xp_arr[0]= Vxp

    exp_x_sq[0] = Vxx + x @ x.T
    exp_p_sq[0] = Vpp + p @ p.T
   
    

    #Force feedback and unfiltered measurement stream estimation 
    if feedback == 'force':
        for i in range(steps-1):
            del_W = np.random.normal(loc=0.0, scale=np.sqrt(del_t), size=(2,1))
            del_Q = 4*eta*k*x*del_t + np.sqrt(2*eta*k)*del_W
            
            
            if not filt:
                p_fb = alpha_p
            else:
                p_fb = p
                
            x += (M_inv@p)*del_t -8*eta*k*(Vxx@x)*del_t + 2*(Vxx@del_Q)
            p += (-K@x) *del_t -8*eta*k* (Vxp.T@x)*del_t + 2*(Vxp.T@del_Q) -(Gamma@p_fb)*del_t
          
    
            Vxx += (M_inv@Vxp.T + Vxp@M_inv - 8*eta*k*(Vxx@Vxx)) * del_t
            Vpp += ((-K@Vxp - Vxp.T@K) +2*k - 8*eta*k*(Vxp@Vxp.T)) * del_t
            Vxp += (M_inv@Vpp - Vxx@K -8*eta*k*(Vxx@Vxp)) * del_t
         
            Vxx = 0.5 * (Vxx + Vxx.T)
            Vpp = 0.5 * (Vpp + Vpp.T)
            
            x_sq = Vxx + x @ x.T
            p_sq = Vpp + p @ p.T
            
            alpha_x = x + (del_W/(np.sqrt(8*eta*k*del_t)))
            
            if i >= 1:
                alpha_p = M@(alpha_x - alpha_x_stream[i].reshape(2,1))/del_t
            else:
                alpha_p = np.zeros((2,1),dtype=float)

            t_now = (i+1) * del_t
            t_arr[i+1] = t_now
            t_per_arr[i+1] = t_now/T
            
            alpha_p_stream[i+1] = alpha_p[:,0]
            alpha_x_stream[i+1] = alpha_x[:,0]
    
           
            exp_x[i+1] = x[:,0]
            exp_p[i+1] = p[:,0]
            var_x_arr[i+1] = Vxx
            var_p_arr[i+1] = Vpp
            cov_xp_arr[i+1] = Vxp
            
            exp_x_sq[i+1] = x_sq
            exp_p_sq[i+1] = p_sq
            
           
    else:
        for i in range(steps-1):
            del_W = np.random.normal(loc=0.0, scale=np.sqrt(del_t), size=(2,1))
            del_Q = 4*eta*k*x*del_t + np.sqrt(2*eta*k)*del_W
            
            if not filt:
                x_fb = alpha_x
                p_fb = alpha_p
            else:
                x_fb = x
                p_fb = p
                
            x += (M_inv@p)*del_t -8*eta*k*(Vxx@x)*del_t + 2*(Vxx@del_Q)
            p += (-K@x) *del_t -8*eta*k* (Vxp.T@x)*del_t + 2*(Vxp.T@del_Q) 
          
    
            Vxx += (M_inv@Vxp.T + Vxp@M_inv - 8*eta*k*(Vxx@Vxx)) * del_t
            Vpp += ((-K@Vxp - Vxp.T@K) +2*k - 8*eta*k*(Vxp@Vxp.T)) * del_t
            Vxp += (M_inv@Vpp - Vxx@K -8*eta*k*(Vxx@Vxp)) * del_t
                
            Vxx = 0.5 * (Vxx + Vxx.T)
            Vpp = 0.5 * (Vpp + Vpp.T)
            
            x_sq = Vxx + x @ x.T
            p_sq = Vpp + p @ p.T
            
            alpha_x = x + (del_W/(np.sqrt(8*eta*k*del_t)))
            
            if i >= 1:
                alpha_p = M@(alpha_x - alpha_x_stream[i].reshape(2,1))/del_t
            else:
                alpha_p = np.zeros((2,1),dtype=float)

            t_now = (i+1) * del_t
            t_arr[i+1] = t_now
            t_per_arr[i+1] = t_now/T
            
            alpha_p_stream[i+1] = alpha_p[:,0]
            alpha_x_stream[i+1] = alpha_x[:,0]
    
           
            exp_x[i+1] = x[:,0]
            exp_p[i+1] = p[:,0]
            var_x_arr[i+1] = Vxx
            var_p_arr[i+1] = Vpp
            cov_xp_arr[i+1] = Vxp
            
            exp_x_sq[i+1] = x_sq
            exp_p_sq[i+1] = p_sq
            


            x_rot = x_fb*(np.cos(w_arr*t_arr[i+1])) - p_fb*(np.sin(w_arr*t_arr[i+1]))
            p_rot = x_fb*(np.sin(w_arr*t_arr[i+1])) + p_fb*(np.cos(w_arr*t_arr[i+1]))
            
            
            phi_rot = np.arctan2(p_rot, x_rot)
            phi_p_rot = np.pi/2 - 2*phi_rot
            phi_p_rot_swapped = phi_p_rot[::-1, :]
            
            w_sq = w_arr**2 * (1 - 2*epsilon*np.cos(2*w_arr*t_arr[i+1] + phi_p_rot) - 2*epsilon_swapped*np.cos(2*w_arr_swapped*t_arr[i+1] + phi_p_rot_swapped))
            
            K = m * np.diag(w_sq.flatten())
            
            
        
    t_scale_vec = np.array([1/w0, 1/w0])
    t_real_arr = t_arr[:, None] * t_scale_vec[None, :]
    
    exp_x_real = exp_x @ Sx.T
    exp_p_real = exp_p @ Sp.T
    
    alpha_x_stream_real = alpha_x_stream @ Sx.T
    alpha_p_stream_real = alpha_p_stream @ Sp.T

    var_x_real = np.empty_like(var_x_arr)
    var_p_real = np.empty_like(var_p_arr)
    cov_xp_real = np.empty_like(cov_xp_arr)
    exp_x_sq_real = np.empty_like(exp_x_sq)
    exp_p_sq_real = np.empty_like(exp_p_sq)
    
    sx0, sx1 = x_scale_vec[0], x_scale_vec[1]
    sp0, sp1 = p_scale_vec[0], p_scale_vec[1]

    for i in range(var_x_arr.shape[0]):
        # Sx @ A @ Sx
        var_x_real[i, 0, 0] = sx0 * sx0 * var_x_arr[i, 0, 0]
        var_x_real[i, 0, 1] = sx0 * sx1 * var_x_arr[i, 0, 1]
        var_x_real[i, 1, 0] = sx1 * sx0 * var_x_arr[i, 1, 0]
        var_x_real[i, 1, 1] = sx1 * sx1 * var_x_arr[i, 1, 1]
    
        # Sp @ A @ Sp
        var_p_real[i, 0, 0] = sp0 * sp0 * var_p_arr[i, 0, 0]
        var_p_real[i, 0, 1] = sp0 * sp1 * var_p_arr[i, 0, 1]
        var_p_real[i, 1, 0] = sp1 * sp0 * var_p_arr[i, 1, 0]
        var_p_real[i, 1, 1] = sp1 * sp1 * var_p_arr[i, 1, 1]
    
        # Sx @ A @ Sp
        cov_xp_real[i, 0, 0] = sx0 * sp0 * cov_xp_arr[i, 0, 0]
        cov_xp_real[i, 0, 1] = sx0 * sp1 * cov_xp_arr[i, 0, 1]
        cov_xp_real[i, 1, 0] = sx1 * sp0 * cov_xp_arr[i, 1, 0]
        cov_xp_real[i, 1, 1] = sx1 * sp1 * cov_xp_arr[i, 1, 1]
    
        # Sx @ A @ Sx
        exp_x_sq_real[i, 0, 0] = sx0 * sx0 * exp_x_sq[i, 0, 0]
        exp_x_sq_real[i, 0, 1] = sx0 * sx1 * exp_x_sq[i, 0, 1]
        exp_x_sq_real[i, 1, 0] = sx1 * sx0 * exp_x_sq[i, 1, 0]
        exp_x_sq_real[i, 1, 1] = sx1 * sx1 * exp_x_sq[i, 1, 1]
    
        # Sp @ A @ Sp
        exp_p_sq_real[i, 0, 0] = sp0 * sp0 * exp_p_sq[i, 0, 0]
        exp_p_sq_real[i, 0, 1] = sp0 * sp1 * exp_p_sq[i, 0, 1]
        exp_p_sq_real[i, 1, 0] = sp1 * sp0 * exp_p_sq[i, 1, 0]
        exp_p_sq_real[i, 1, 1] = sp1 * sp1 * exp_p_sq[i, 1, 1]
    return (t_real_arr, t_per_arr,exp_x_real,exp_p_real, alpha_x_stream_real, alpha_p_stream_real, var_x_real, var_p_real, cov_xp_real, exp_x_sq_real,exp_p_sq_real)

def plot_sme_traj_2D(del_t,period_number, eta, k, gamma, x0, p0, feedback, fx = 5e4, fy = 2e4, filt=None, M=None, plot_bool = False):
    t_real_arr, t_per_arr,exp_x_real,exp_p_real, alpha_x_stream_real, alpha_p_stream_real, var_x_real, var_p_real, cov_xp_real,exp_x_sq_real,exp_p_sq_real = sme_evolution_2D(del_t=del_t, period_number=period_number, eta=eta, k=k, gamma=gamma, x0=x0, p0=p0, feedback=feedback,fx = fx, fy = fy, filt=filt)
    if not filt:
        filt_tag = "Measurement Stream"
    elif filt == "kalman":
        filt_tag = "Kalman Filter"
    if feedback == "force":
        feed_tag = "Force Feedback"
    elif feedback == "parametric":
        feed_tag = "Parametric Feedback"
    dimensions = ["x","y"]
    
    save_dir = f"plots/{filt_tag}_{feed_tag}"
    os.makedirs(save_dir, exist_ok=True)
    
    for i,dim in enumerate(dimensions):
        if plot_bool:
            plt.figure()
            plt.plot(t_real_arr[:,i]*10**6, exp_x_real[:,i])
            plt.xlabel("t ($\\mu s$)")
            plt.ylabel("<x> (m)")
            plt.title(f"{filt_tag} {feed_tag} {dim} Position Expectation Value Time Evolution")
            plt.show()
            
            plt.figure()
            plt.plot(t_real_arr[:,i]*10**6, exp_p_real[:,i])
            plt.xlabel("t ($\\mu s$)")
            plt.ylabel("<p> ($kg \\cdot m \\cdot s^{-1}$)")
            plt.title(f"{filt_tag} {feed_tag} {dim} Momentum Expectation Value Time Evolution")
            plt.show()
            
            plt.figure()
            plt.plot(t_real_arr[:,i]*10**6, alpha_x_stream_real[:,i])
            plt.xlabel("t ($\\mu s$)")
            plt.ylabel(r"$\alpha_{x}$ (m)")
            plt.title(f"{filt_tag} {feed_tag} {dim} Experiment Position Stream Time Evolution")
            plt.show()
        else:
            plt.figure()
            plt.plot(t_real_arr[:,i]*10**6, exp_x_real[:,i])
            plt.xlabel("t ($\\mu s$)")
            plt.ylabel("<x> (m)")
            plt.title(f"{filt_tag} {feed_tag} {dim} Position Expectation Value Time Evolution")
            filename = f"{save_dir}/eta{eta}_k{k}_gamma{gamma}_{dim}_x.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            plt.figure()
            plt.plot(t_real_arr[:,i]*10**6, exp_p_real[:,i])
            plt.xlabel("t ($\\mu s$)")
            plt.ylabel("<p> ($kg \\cdot m \\cdot s^{-1}$)")
            plt.title(f"{filt_tag} {feed_tag} {dim} Momentum Expectation Value Time Evolution")
            filename = f"{save_dir}/eta{eta}_k{k}_gamma{gamma}_{dim}_p.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            
            plt.figure()
            plt.plot(t_real_arr[:,i]*10**6, alpha_x_stream_real[:,i])
            plt.xlabel("t ($\\mu s$)")
            plt.ylabel(r"$\alpha_{x}$ (m)")
            plt.title(f"{filt_tag} {feed_tag} {dim} Experiment Position Stream Time Evolution")
            filename = f"{save_dir}/eta{eta}_k{k}_gamma{gamma}_{dim}_alpha_x.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
def t_eff(p_sq, x_sq, m, w):
    kB = 1.3806*10**-23
    
    T_eff = (p_sq/m + m*w**2*x_sq)/(2*kB)
    return T_eff

feedback_style_dict = {"force": {"color": "tab:blue","label": "Force Feedback"},"parametric": {"color": "tab:orange","label": "Parametric Feedback"}}

filter_style_dict = {None: {"linestyle": "-","label": "Measurement Stream"},"kalman": {"linestyle": "--","label": "Kalman Filter"}}

def effective_temp_plot(del_t,period_number, eta, k, gamma_list, x0, p0, feedback, filt=None, m_bead=4.8e-18,fx = 5e4, fy = 2e4, M= None, plot_bool = False):
    Tx_list = []
    Ty_list = []
    w_real=[fx*2*np.pi, fy*2*np.pi]
    for gamma in gamma_list:
        t_real_arr, t_per_arr,exp_x_real,exp_p_real, alpha_x_stream_real, alpha_p_stream_real, var_x_real, var_p_real, cov_xp_real,exp_x_sq_real,exp_p_sq_real = sme_evolution_2D(del_t=del_t, period_number=period_number, eta=eta, k=k, gamma=gamma, x0=x0, p0=p0, fx = fx, fy = fy, feedback=feedback, filt=filt, M=None)
        dimensions = ["x","y"]
        for i,dim in enumerate(dimensions):

            steps = len(t_real_arr[:,i])
            average_increment = int(steps/4)
            exp_x_sq_dim = exp_x_sq_real[:,i]
            exp_p_sq_dim = exp_p_sq_real[:,i]
            x_sq_real = np.mean(exp_x_sq_dim[-average_increment:])
            p_sq_real = np.mean(exp_p_sq_dim[-average_increment:])
            t_e = t_eff(p_sq_real, x_sq_real, m=m_bead, w=w_real[i])
            if i == 0:
                Tx_list.append(t_e)
            else:
                Ty_list.append(t_e)
                
    
    T_list = [Tx_list, Ty_list]
    feedback_style = feedback_style_dict[feedback]
    filter_style = filter_style_dict[filt]
    label = (f"{feedback_style['label']}, "f"{filter_style['label']}, "+ rf"$\eta={eta}$, $k={k}$")    

    if plot_bool:
        for i,dim in enumerate(dimensions):
            gamma_list_real = gamma_list* (w_real[i]/(2*np.pi))
            fig, ax = plt.subplots()
            ax.plot(gamma_list_real,np.array(T_list[i]),color=feedback_style["color"],linestyle=filter_style["linestyle"],marker="o",label=label)
        
            ax.set_xlabel(r"$\Gamma$ (Hz)")
            ax.set_ylabel(f"T_{dim} (K)")
            ax.set_title(f"{dim} Effective Temperature vs. Damping Rate")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend()
            plt.show()
    
    return T_list

def effective_temp_plot_degen(del_t,period_number, eta, k, gamma, x0, p0, feedback, filt=None, m_bead=4.8e-18,fx = 5e4, fy = 2e4, M= None, plot_bool = False):
    Tx_list = []
    Ty_list = []
    ratio_list= []
    pairs = [
    (np.sqrt(fx*fy) * (fx/fy)**((1-s)/2),
     np.sqrt(fx*fy) / (fx/fy)**((1-s)/2))
    for s in np.linspace(0,1,10)]
    
    for pair in pairs:
        fx, fy = pair
        ratio_list.append(fx/fy)
        w_real = [fx*2*np.pi, fy*2*np.pi]
        t_real_arr, t_per_arr,exp_x_real,exp_p_real, alpha_x_stream_real, alpha_p_stream_real, var_x_real, var_p_real, cov_xp_real,exp_x_sq_real,exp_p_sq_real = sme_evolution_2D(del_t=del_t, period_number=period_number, eta=eta, k=k, gamma=gamma, x0=x0, p0=p0, fx = fx, fy = fy, feedback=feedback, filt=filt, M=None)
        dimensions = ["x","y"]
        for i,dim in enumerate(dimensions):

            steps = len(t_real_arr[:,i])
            average_increment = int(steps/4)
            exp_x_sq_dim = exp_x_sq_real[:,i]
            exp_p_sq_dim = exp_p_sq_real[:,i]
            x_sq_real = np.mean(exp_x_sq_dim[-average_increment:])
            p_sq_real = np.mean(exp_p_sq_dim[-average_increment:])
            t_e = t_eff(p_sq_real, x_sq_real, m=m_bead, w=w_real[i])
            if i == 0:
                Tx_list.append(t_e)
            else:
                Ty_list.append(t_e)
                
    
    T_list = [Tx_list, Ty_list]
    feedback_style = feedback_style_dict[feedback]
    filter_style = filter_style_dict[filt]
    label = (f"{feedback_style['label']}, "f"{filter_style['label']}, "+ rf"$\eta={eta}$, $k={k}$")    

    if plot_bool:
        
        for i,dim in enumerate(dimensions):
            
            fig, ax = plt.subplots()
            ax.plot(np.array(ratio_list),np.array(T_list[i]),color=feedback_style["color"],linestyle=filter_style["linestyle"],marker="o",label=label)
        
            ax.set_xlabel(r"wx/wy")
            ax.set_ylabel(f"T_{dim} (K)")
            ax.set_title(f"{dim} Effective Temperature vs. Damping Rate")
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
        