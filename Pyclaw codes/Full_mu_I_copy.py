#!/usr/bin/env python
# encoding: utf-8

r"""
Burgers' equation
=========================

Solve the inviscid Burgers' equation:

.. math:: 
    q_t + \frac{1}{2} (q^2)_x = 0.

This is a nonlinear PDE often used as a very simple
model for fluid dynamics.

The initial condition is sinusoidal, but after a short time a shock forms
(due to the nonlinearity).
"""
from __future__ import absolute_import
import numpy as np
from clawpack import riemann 
import time

start1 = time.time()
def particle_fraction(state):
    """ Compute phi from q (mu_w)  and store in state.p."""
    import numpy as np
    from numpy import log
    from scipy.optimize import fsolve,root
    
    q = state.q[0,:]
    R = state.aux[2,:]
    mu1 = state.problem_data['mu1']
    D = state.problem_data['D']
    phim = state.aux[0,:]
    
    num_rp = state.q.shape[1]
    m = np.empty( (1, num_rp) )
    F = np.empty( (1, num_rp) )
    F_4 = np.empty( (1, num_rp) )


    def F1(m1,mw):
      #F1 = (4/(3*mw^2))*( (1 - m1 + mw)**(1/2)*(2*m1 - 2 + mw) - (m1-2) + (3/2)*m1**2 )
      
      F1 = (m1)/mw**2 - 1/mw + (2*(-(m1 - mw))**(3/2))/(3*mw**2) + (2*(-(m1 - mw))**(1/2)*(m1 + 1))/mw**2  -(2*log((-(m1 - mw))**(1/2\
      ) + 1)*(m1 + 1))/mw**2 + (m1/mw)**2

      return(F1)

    def F2(mu1,muw):
        #F = 1/(14*muw**4) - (mu1**2 - 2*mu1*muw + 2*mu1 + muw**2 - 3)/(6*muw**4) - ((muw - mu1 + 1)**(7/2))/(14*muw**4) + (mu1 - 3)/(10*muw**4) - ((muw\
        #-mu1+1)**(5/2)*(mu1 - 3))/(10*muw**4) + ((muw - mu1 + 1)**(3/2)*(mu1**2 - 2*mu1*muw + 2*mu1 + muw**2 - 3))/(6*muw**4) + ((mu1 - 1)*(- mu1**2+ \
        #2*mu1*muw-muw**2  + 1))/(2*muw**4) - ((muw - mu1 + 1)**(1/2)*(mu1 - 1)*(- mu1**2 + 2*mu1*muw - muw**2 + 1))/(2*muw**4)
        
        F = ((- mu1**2 + 2*mu1*muw + mu1 - muw**2 + 1))/(4*muw**3) - ((mu1 - muw)**3)/(12*muw**4) - ((-(mu1 - muw))**(7/2))/(14*muw**4) - ((-(mu1 -muw))**(3/2\
        )*(- mu1**2 + 2*mu1*muw + mu1 - muw**2 + 1))/(6*muw**4) - ((-(mu1 - muw))**(5/2)*(mu1 + 1))/(10*muw**4) + ((mu1 - muw)**2*(mu1 + 1))/(8*muw**4) - (\
        mu1*(- mu1**2 + 2*mu1*muw + mu1 - muw**2 + 1))/(4*muw**4) + (log((-(mu1 - muw))**(1/2) + 1)*(mu1 + 1)*(- mu1**2 + 2*mu1*muw - muw**2 +1))/(2*muw**4)\
        - ((-(mu1 - muw))**(1/2)*(mu1 + 1)*(- mu1**2 + 2*mu1*muw - muw**2 + 1))/(2*muw**4) 
        
        F2 = (mu1/(2*muw))**2*(1 - mu1/muw)**2 + 2*F
        return(F2)

    
    
    def F3(mu1,muw):
        F3 = ((muw-mu1)**2/(4*muw**4))*(3*muw**2 + 2*mu1*muw - 4*mu1**2)
        return(F3)
      

    def F4(mu1,muw,pm):
        F = (3*(mu1 - 1))/muw + (6)/(5*muw**2*pm) - 1/(3*muw**2*pm**2) - (3*mu1*(mu1 - 1))/muw**2 + (2*pm*(mu1 - 1))/muw**2-(6*((muw-mu1+1))**(5/2))/(5\
        *muw**2*pm) + ((muw - mu1 + 1)**3)/(3*muw**2*pm**2) - ((3*pm**2 + mu1 - 1))/(2*muw**2*pm**2) + (2*(pm**2 + 3*mu1 - 3))/(3*muw**2*pm) - (2*((muw-\
        mu1 + 1))**(3/2)*(pm**2 + 3*mu1 - 3))/(3*muw**2*pm) + ((muw - mu1 + 1)**2*(3*pm**2 + mu1 - 1))/(2*muw**2*pm**2) - (2*pm*((muw - mu1 + 1))**(1/2)\
        *(mu1 - 1))/muw**2
        mu=1
        F4 = (1/mu)*((1-pm)**3/(pm**2)*(mu1/muw) + 2*F)

        

        return(F4)
      
    
    def func(mw,*arg):
        
        mu1,R,phim,q = arg

        return ( q/((R**2)*phim) - F1(mu1,mw) )
        #(4/(3*mw**2))*( (1 - mu1 + mw)**(1/2)*(2*mu1 - 2 + mw) - (mu1-2) + (3/2)*mu1**2) )

    x_in = 1

    for x in range(num_rp):
        arg = (mu1,R[x], phim[x], q[x])
        sol = root(func,x_in, args = arg)
        m[0,x] = sol.x
        x_in = m[0,x]


    #state.problem_data['mu1'] = .4
    phi = q/(R**2)
    F_2 = F2(mu1,m)
    F_3 = F3(mu1,m)
    F_4 = F4(mu1,m,phim)
    Flux = (phim*F_2)/(F_3 + D*F_4/(R**2))
    
    state.p[0,:] = phi
    state.p[1,:] = m
    state.p[2,:] = Flux
    
    

    

def setup(use_petsc=0,kernel_language='Python',outdir='./_Copy',solver_type='classic',disable_output=False):

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if kernel_language == 'Python': 
        
        riemann_solver = riemann.Mu_I_copy.Full_model_copy
        
    elif kernel_language == 'Fortran':
        
        riemann_solver = riemann.Mu_I_copy.Full_model_copy
        


    if solver_type=='sharpclaw':
        solver = pyclaw.SharpClawSolver1D(riemann_solver)
    else:
        solver = pyclaw.ClawSolver1D(riemann_solver)
        solver.limiters = pyclaw.limiters.tvd.vanleer
        #solver.cfl_max = 0.6
        #solver.cfl_desired = 0.3
        #solver.step_source = step_Source

    
    solver.kernel_language = kernel_language
        
    solver.bc_lower[0] = pyclaw.BC.extrap
    solver.bc_upper[0] = pyclaw.BC.extrap
    solver.aux_bc_lower[0]=pyclaw.BC.extrap
    solver.aux_bc_upper[0]=pyclaw.BC.extrap
    solver.max_steps = 20000 
    solver.verbosity = 4

    x = pyclaw.Dimension(-0.5,1.5,8000,name='x')
    domain = pyclaw.Domain(x)
    num_eqn = 1
    num_aux=3
    state = pyclaw.State(domain,num_eqn,num_aux)
    #state.mp = 5

  
    
    c = 0.2
    d = 5

    xc = state.grid.x.centers
    #state.q[0,:] = .2 #+ .2*np.exp(-((xc-.5)*5)**2)
    #state.q[0,:] = .4
    state.problem_data['efix']=False
    state.aux[0,:] = 0.9 - c*np.exp(-((xc-.6)*d)**2)            #phi_m
    state.aux[1,:] = 2*c*d**2*np.exp(-((xc-.6)*d)**2)*(xc-.6) #phi_mz
    #state.aux[0,:] = 1                                          #phi_m
    #state.aux[1,:] = 0                                          #phi_mz
    state.aux[2,:] = 1 #- .2*np.exp(-((xc-.6)*5)**2)             #R(z)
     
    state.q[0,:] = 0.4 * (state.aux[2,:])**2                    #q = phi* R^2 
    
 
    state.problem_data['D'] = .001
    state.problem_data['mu1'] = 0.5
    
    claw = pyclaw.Controller()
    claw.tfinal = 0.01
    claw.solution = pyclaw.Solution(state,domain)
    claw.verbosity = 4

    claw.solver = solver
    claw.outdir = outdir
    claw.setplot = setplot
    claw.keep_copy = True
    claw.num_output_times=5
    claw.output_style =1
    claw.write_aux_always = True
    claw.overwrite = True
    

    if disable_output:
        claw.output_format = None
    #claw.compute_p = particle_fraction
    claw.write_aux_init = False
    #claw.file_prefix_p = 'fort'

    return claw

def step_Source(solver,state,dt):
    """
    Geometric source terms for Euler equations with cylindrical symmetry.
    Integrated using a 2-stage, 2nd-order Runge-Kutta method.
    This is a Clawpack-style source term routine, which approximates
    the integral of the source terms over a step.
    """
    

    q = state.q
    phim = state.aux[0,:]
    phimz = state.aux[1,:]
    R = state.aux[2,:]
    D = state.problem_data['D']
    
    f = q**2/(phim - q)**2
    f_phim = -2*q**2/(phim - q)**3
    f_phi  = 2*q*phim/(phim - q)**3 
    
    H = np.empty(q.shape)
    H2 = np.empty(q.shape)
    qstar = np.empty(q.shape)
          
    dt2 = dt/2.
    H[0,:] = phimz*q/(phim*(R**2*q+D)) 
    qstar[0,:] = q[0,:] + dt2*H[0,:]
    
    H2[0,:] = phimz*qstar/(phim*(R**2*qstar+D))    
    q[0,:] = q[0,:] + dt * H2[0,:]
    
    





def setplot(plotdata):
    """ 
    Plot solution using VisClaw.
    """ 
    plotdata.clearfigures()  # clear any old figures,axes,items data

    # Figure for q[0]
    plotfigure = plotdata.new_plotfigure(name='q[0]', figno=0)


    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = [0, 3]
    plotaxes.title = 'q[0]'
    plotaxes.afteraxes = add_true_solution
    #plotaxes.afteraxes = label_axes
    #plotaxes.afteraxes = add_title
  

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = 0
    plotitem.plotstyle = '-'
    plotitem.color = 'b'
    plotitem.kwargs = {'linewidth': .5}
    
   # plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    #import os
    #plotitem.outdir = os.path.join(os.getcwd(), '_p')
    #plotitem.plot_var = 0
    #plotitem.plotstyle = '-+'
    #plotitem.color = 'r'

    
    return plotdata

    
def add_true_solution(current_data):
   import matplotlib.pyplot as plt
   from pylab import sin, plot, title
   phim = current_data.aux
   x = current_data.x
   t = current_data.t
   q = current_data.q
   aux = current_data.aux
   phim = aux[0,:]
   R = aux[2,:]
   
   
   phi = np.empty(q.shape)
   
   
   phi = q[0,:]/(R**2)
   #q = current_data.plotdata.p
   
   
   #plot(x,sin(x),'r')
   plot(x,phim,'r')
   plot(x,R,'b')
   plot(x,phi,'g' )
   plt.xlabel(r'$z$')
   plt.ylabel(r'$\overline{\phi}$')
   title("Solution at time t = %10.4e" % t, fontsize=10)
   
   
   
   
def add_title(current_data):
    from pylab import title
    t = current_data.t
    title("Solution at time t = %10.4e" % t, fontsize=20)
    

def label_axes(current_data):
    import matplotlib.pyplot as plt
    plt.xlabel('z')
    plt.ylabel('r')
    


if __name__=="__main__":
    #print('Line 95')
    #x0=2
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup,setplot)
    end1 = time.time()
    print('Time taken by the Main program  :', end1 - start1)
