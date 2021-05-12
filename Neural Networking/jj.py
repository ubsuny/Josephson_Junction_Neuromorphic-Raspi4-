"""
This python file holds all the functions used in the projects.
Currently holds the function for calculating the time evolution of
a single junction and for a single time step of a junction both based
on Euler's method
"""
import numpy as np

def junction(t_0, t_f, num, damp, i):
    """
    Calculates the phase and change in phase as a function
        of time from initial time, t0, to final time, tf, in [num] steps
         with parameters i (current) and damp (damping)
    """
    t_s = (t_f - t_0) / num
    t_span = np.arange(t_0, t_f + t_s, t_s)  # time span

    phi = np.zeros_like(t_span)
    phi_dot = np.zeros_like(t_span)
    # Initial Conditions
    phi[0] = 0
    phi_dot[0] = 0

    for j in range(1, num + 1):
        phi[j] = phi[j - 1] + phi_dot[j - 1] * t_s
        phi_dot[j] = phi_dot[j - 1] + (i - np.sin(phi[j - 1]) - damp * phi_dot[j - 1]) * t_s
    return phi, phi_dot, t_span

def junction_step(p_0, pd_0, dt, damp, i):
    """
    Calculates a single timestep of phase and change in phase given the
    current values p0 and v0, length of time ts, and parameters i
     (current) and damp (damping) with Newton's method
    """
    phi = p_0 + pd_0 * dt
    phi_dot = pd_0 + (i - np.sin(p_0) - damp * pd_0) * dt

    return phi, phi_dot

def currents(lmda,phi_p,phi_c,lmda_s,i,lmda_p,i_b,eta):
    """
    Calculate the pulse and control current inside a junction
    """
    i_p = -lmda*(phi_c + phi_p) + lmda_s*i + (1-lmda_p)*i_b
    i_c = (-lmda*(phi_c + phi_p) + lmda_s*i - lmda_p*i_b)/eta
    
    return i_p, i_c

def change_weights(A,B,tau,t1,t0):
    x = t1 - t0
    if x > 0:
        dw = A*np.exp(-x/tau)
    else:
        dw = -B*np.exp(x/tau)
    return dw
    

def synapse_step(v_0, vd_0, i_0, id_0, v1p, v2p, v2c, gamma, omega, Q, lmda, lmda_syn, r12, dt):
    """
    Calculates a single timestep of output voltage and current given the
    outputs of the previous neuron
    
    """
    v = v_0 + vd_0 * dt
    v_dot = vd_0 + omega ** 2 * (v1p - Q*omega*lmda_syn/lmda * i_0
                                 - lmda_syn/lmda * id_0 - v_0 - Q/omega * vd_0)*dt
    
    i = i_0 + id_0 * dt
    i_dot = (v_0 - lmda_syn*(v2c + v2p) - r12/gamma*i_0) * lmda/(lmda_syn*(1-lmda_syn))
    
    return v, v_dot, i, i_dot
