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

def junction_step(p_0, pd_0, t_s, damp, i):
    """
    Calculates a single timestep of phase and change in phase given the
    current values p0 and v0, length of time ts, and parameters i
     (current) and damp (damping)
    """
    phi = p_0 + pd_0 * t_s
    phi_dot = pd_0 + (i - np.sin(p_0) - damp * pd_0) * t_s

    return phi, phi_dot
