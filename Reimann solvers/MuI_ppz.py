#!/usr/bin/env python
# encoding: utf-8
r"""
Riemann solvers for Burgers equation

.. math::
    u_t + \left ( \frac{1}{2} u**2 \right)_x = 0

:Authors:
    Kyle T. Mandli (2009-2-4): Initial version
"""
# ============================================================================
#      Copyright (C) 2009 Kyle T. Mandli <mandli@amath.washington.edu>
#
#  Distributed under the terms of the Berkeley Software Distribution (BSD) 
#  license
#                     http://www.opensource.org/licenses/
# ============================================================================

from __future__ import absolute_import
num_eqn = 1
num_waves = 3

import numpy as np
import cmath as cm
from numpy import pi, real,log
#from cmath import log
from scipy.optimize import fsolve,root
from scipy.integrate import solve_ivp, odeint
from  matplotlib import pyplot  



def Full_ppz(q_l,q_r,aux_l,aux_r,problem_data):
    r"""
    Riemann solver for Burgers equation in 1d
         
    *problem_data* should contain -
     - *efix* - (bool) Whether a entropy fix should be used, if not present, 
       false is assumed
    
    See :ref:`pyclaw_rp` for more details.
    
    :Version: 1.0 (2009-2-4)
    """
    
    
    num_rp = q_l.shape[1]
    #x0 = q_l.shape[1]
    m_ml = np.empty( (num_eqn, num_rp) )
    m_mr = np.empty( (num_eqn, num_rp) )
    m_pl = np.empty( (num_eqn, num_rp) )
    m_pr = np.empty( (num_eqn, num_rp) )
    m_rl = np.empty( (num_eqn, num_rp) )
    m_rr = np.empty( (num_eqn, num_rp) )

    pp_ml = np.zeros( (num_rp) )
    pp_mr = np.zeros( ( num_rp) )
    pp_pl = np.zeros( (num_rp) )
    pp_pr = np.zeros( (num_rp) )
    pp_rl = np.zeros( (num_rp) )
    pp_rr = np.zeros( (num_rp) )
    pz_ml = np.zeros( (num_rp) )
    pz_mr = np.zeros( ( num_rp) )
    pz_pl = np.zeros( (num_rp) )
    pz_pr = np.zeros( (num_rp) )
    pz_rl = np.zeros( (num_rp) )
    pz_rr = np.zeros( (num_rp) )

    # Output arrays
    wave = np.empty( (num_eqn, num_waves, num_rp) )
    s = np.empty( (num_waves, num_rp) )
    amdq = np.empty( (num_eqn, num_rp) )
    apdq = np.empty( (num_eqn, num_rp) )

    # Basic solve
    
    # dFlux/dz = dFlux/dphim phim_z + dFlux/dR R_z + dFlux/dmu_w muw_z
    
    q=.5 * (q_r[0,:] + q_l[0,:])
    phim = .5*(aux_l[0,:] + aux_r[0,:])
    R = .5*(aux_l[2,:] + aux_r[2,:])
    
    phim_r = aux_r[0,:]
    phim_l = aux_l[0,:]
    R_r = aux_r[2,:]
    R_l = aux_l[2,:]
    
    
    D = problem_data['D']
    mu1 = problem_data['mu1']
    # z0 = problem_data['z0']
    # z01 = problem_data['z1']

    # x0_l = 0.5*np.ones((1,num_rp))
    # x0_r = 0.5*np.ones((1,num_rp))

    #print((Full_mu_I.x0_l))


    #xl = Full_mu_I.x0_l
    #xr = Full_mu_I.x0_r
    # delx = (1.5+0.5)/(num_rp-3)
    # #print(num_rp)
    # z = np.linspace(-0.5-delx,1.5+delx,num_rp)
    # pyplot.figure(15)
    # pyplot.plot(z,q_r[0,:] - q_l[0,:])
    # #pyplot.plot(z1,Fr_l[0,1:-1],'b',linestyle='dashed')
    # pyplot.savefig('plot_Qdiff_rl.png',dpi = 400)
    # pyplot.figure(16)
    # pyplot.plot(z,q_r[0,:])
    # #pyplot.plot(z1,Fr_l[0,1:-1],'b',linestyle='dashed')
    # pyplot.savefig('plot_QR.png',dpi = 400)
    
    def F1(mu,mw):
      
      #F1 = (4/(3*mw**2))*( (1 - m1 + mw)**(1/2)*(2*m1 - 2 + mw) - (m1-2) + (3/2)*m1**2 )
        #mu=m1


        if(mw>mu):
            F1 = 2*(mw*(-3 + 2*(mw - mu)**(1/2)) + 3*mu + 2*(mw - mu)**(1/2)*(3 + 2*mu) - 6*(1 + mu)*log(1 + (mw - mu)**(1/2)))/(3*mw**2) + (mu/mw)**2
        elif(mw==mu):
            F1 = mw**0
        else:
            F1 = 1000

        return(F1)


    
    def func(mw,*arg):
        
        mu1,R,phim,q = arg

        return ( q/((R**2)*phim) - F1(mu1,mw) )
        #(4/(3*mw**2))*( (1 - mu1 + mw)**(1/2)*(2*mu1 - 2 + mw) - (mu1-2) + (3/2)*mu1**2) )
    
    def F2(mu1,muw):

        m = muw
        rc = mu1/muw

        mu=mu1
        mw = muw
        F = (2*mu - mw)/(2*mw**2) - (mw - mu)**(1/2)/(7*mw) - (mu**2 - 1)/(2*mw**3) - mu/(2*mw**2) + 1/(6*mw) + (mu + 1)/(4*mw**2) - mu**3/(6*mw**4) - ((mw - mu)**(1/2)*(- 64*mu**3 + 140*mu**2*mw - 119*mu**2 - 70*mu*mw**2 + 210*mu*mw + 70*mu - 105*mw**2 + 105))/(105*mw**4) + ((8*mu - 7)*(mw - mu)**(1/2))/(35*mw**2) - (mu*(- mu**2 + 2*mu*mw - mw**2 + 1))/(2*mw**4) + (mu**2*(mu - 1))/(4*mw**4) + ((mw - mu)**(1/2)*(32*mu**2 - 70*mu*mw + 7*mu + 35*mw**2 - 35))/(105*mw**3) + (mu**2*(mu - mw)**2)/(4*mw**4)  + (log((-mw + mu + 1)*(mw - mu + 2*(mw - mu)**(1/2) + 1)/(mu - mw + 1))*(mu + 1)*(- mu**2 + 2*mu*mw - mw**2 + 1))/(2*mw**4) #+ (log((mw - mu + 2*(mw - mu)**(1/2) + 1)/(mu - mw + 1))*(mu + 1)*(mu - mw + 1)*(mw - mu + 1))/(2*mw**4)  
        
        FF = np.where(F<0,0,F)
        F2 = np.where(mw>=mu,FF,0)
    
        return(F2)
    
    
    
    def F3(m1,mw):
        
        rc = m1/mw

        #F3 = ((muw-mu1)**2/(4*muw**4))*(3*muw**2 + 2*mu1*muw - 4*mu1**2)
        F = (rc**2*(rc - 1/2))/2 - rc/6 + (rc**2*(rc - 1)**2)/4 - (5*rc**4)/24 + 1/8
        FF = np.where(F<0,0,F)
        F3 = np.where(mw>=m1,FF,0)
        return(F3)
      
    
    def F4(mu,mw,pm):
        m=mw
        rc = mu1/mw
        F = -(16*mu**2*(mw - mu)**(1/2) - 24*mw**2*(mw - mu)**(1/2) + 60*pm**3*(mw - mu)**(1/2) + 15*mu*mw**2 + 30*mu*pm**3 + 45*mw**2*pm - 30*mw*pm**3 - 5*mu**3 - 15*mw**2 - 10*mw**3 - 30*pm**3*log((mw - mu + 2*(mw - mu)**(1/2) + 1)) + 15*mu**2*pm**3 - 45*mw**2*pm**2 + 8*mu*mw*(mw - mu)**(1/2) - 30*mu*pm**3*log((mw - mu + 2*(mw - mu)**(1/2) + 1)) - 24*mu**2*pm*(mw - mu)**(1/2) + 40*mu*pm**3*(mw - mu)**(1/2) + 36*mw**2*pm*(mw - mu)**(1/2) + 20*mw*pm**3*(mw - mu)**(1/2) - 12*mu*mw*pm*(mw - mu)**(1/2))/(15*mw**2*pm**2)
 
        F4 = np.where(mw>mu,F,(1 - pm)**3/(pm**2))

        #Kphi = 1/phi^2
        #F = (2*m)/3 - m*rc - (2*m*rc**3)/3 + 2*rc**2*((m*rc)/2 - 1/2) + rc**2 + 4*m**(1/2)*((2*rc*(1 - rc)**(3/2))/3 + (2*(1 - rc)**(5/2))/5) + 1
        #F4 = np.where(mw>mu,F,(1/(pm**2)))

        return(F4)
      
    def ppz_val(pp,m1,mw,D,R,pm):

        #ppz_val = (1/(D*F4(m1,muw,pm)))*(1/(R**2) - 2*muw*R*pp*F3(m1,muw) ) - 2*muw*pp/(R)
        e =1/(1e-3)
        
        ppz_val = (1/(D*F4(m1,mw,pm))*(1/(R**2)-2*mw*R*pp*F3(m1,mw))- 2*mw*pp/(R))
        # pyplot.figure(900)
        # pyplot.plot(z,1/(D*F4(m1,mw,pm))*(1/(R**2))- 2*mw*pp/(R),linewidth=0.1)
        # pyplot.savefig('plot_ppval_1.png',dpi = 300)
        # pyplot.figure(901)
        # a=pp*F3(m1,mw)
        # pyplot.plot(z,1-mw*pp*F3(m1,mw),linewidth=0.1)
        
        # pyplot.savefig('plot_ppval_2.png',dpi = 300)
        #print(muw)
        return(ppz_val)
    
    def Ppz(t,y1,*arg):
    
        m1,mw,D,R,pm = arg
        
        Ppz = ((1/(D*F4(m1,mw,pm)))*(1/(R**2) - 2*mw*R*y1*F3(m1,mw) ) - 2*mw*y1/(R))
        #Ppz = 1/(D*F4(m1,mw,pm))*(1/(R**2)- 2*mw*R*y*F3(m1,mw))- 2*mw*y/(R)
        return(Ppz)
    
    def Pp(m1,mw,D,R,pm):

        Pp = 1/(2*mw*R**3*(F3(m1,mw) + D*F4(m1,mw,pm)/(R**2)))

        return(Pp)

                

    
    
    x_l=4
    x_r=4


    for x in range(num_rp):
        
        # to get flux_mu mu_w
        arg = (mu1,R[x], phim[x], q_l[0,x] )
        sol = root(func,x_l, args = arg)
        m_ml[0,x] = sol.x
        x_l = m_ml[0,x]


        arg = (mu1,R[x], phim[x], q_r[0,x])
        if(x==1):
            sol = root(func,x_l, args = arg)
            m_mr[0,x] = sol.x
            x_r = m_mr[0,x]
        else:
            sol = root(func,x_r, args = arg)
            m_mr[0,x] = sol.x
            x_r = m_mr[0,x]
        
        # to get flux_pm mu_w
        arg = (mu1,R[x], phim_l[x], q[x] ) 
        sol = root(func,x_l, args = arg)
        m_pl[0,x] = sol.x
        
            #arg = (mu1,R_r[x], phim_r[x], q_r[0,x])
        arg = (mu1,R[x], phim_r[x], q[x])
        sol = root(func,x_r, args = arg)
        m_pr[0,x] = sol.x

        # to get flux_R mu_w
        arg = (mu1,R_l[x], phim[x], q[x] )
        sol = root(func,x_l, args = arg)
        m_rl[0,x] = sol.x
        

        arg = (mu1,R_r[x], phim[x], q[x])
        sol = root(func,x_r, args = arg)
        m_rr[0,x] = sol.x


    #Calculating Pp by solving ode Ppz = f(mw,m1,..Pp)
    
    delx = (1.5 + 0.5)/(num_rp-3)    
    z = np.linspace(-0.5-delx,1.5+delx,num_rp)

    #pyplot.figure(7)
    #pyplot.plot(z,F3(mu1,m_ml[0,:]))
    #pyplot.savefig('plot_m.png',dpi = 300)

    # #raise Exception('Inside Riemann solver')
    pp_ml[0] = Pp(mu1,m_ml[0,0],D,R[0],phim[0])
    #pp_ml[1] = Pp(mu1,m_ml[0,1],D,R[1],phim[1]) 
    pp_mr[0] = Pp(mu1,m_mr[0,0],D,R[0],phim[0])
    pp_pl[0] = Pp(mu1,m_pl[0,0],D,R[0],phim_l[0])
    #pp_pl[1] = Pp(mu1,m_pl[0,1],D,R[1],phim_l[1])
    pp_pr[0] = Pp(mu1,m_pr[0,0],D,R[0],phim_r[0])
    pp_rl[0] = Pp(mu1,m_rl[0,0],D,R_l[0],phim[0])
    #pp_rl[1] = Pp(mu1,m_rl[0,1],D,R_l[1],phim[1])
    pp_rr[0] = Pp(mu1,m_rr[0,0],D,R_r[0],phim[0])
    #print(R_r[0])
    pp_mlI = pp_ml[0]
    pp_mrI = pp_mr[0]
    pp_plI = pp_pl[0]
    pp_prI = pp_pr[0]
    pp_rlI = pp_rl[0]
    pp_rrI = pp_rr[0]

    # # pp_ml[1]=pp_ml[0]
    # # pp_pl[1]=pp_pl[0]
    # # pp_rl[1]=pp_rl[0]

    x=0
    for x in range(1, num_rp):
        #print(x)
        #if (x<(num_rp-1)):
        arg = (mu1,m_ml[0,x],D,R[x],phim[x])
        sol = solve_ivp( Ppz, [z[x-1], z[x]], [pp_mlI] , t_eval=[ z[x]],args = arg, rtol = 1e-7,atol = 1e-9,method = 'LSODA')
        X = sol.y
        pp_ml[x] = X[0,0]
        pp_mlI = X[0,0]
        #print(sol.y) 
        #print(pp_ml[x])
        #raise Exception('ODE solved')
        #try:
        arg1 = (mu1,m_mr[0,x],D,R[x],phim[x])
        sol1 = solve_ivp(Ppz, [z[x-1], z[x]], [pp_mrI], t_eval=[z[x]],args = arg1, rtol = 1e-7,atol = 1e-9,method = 'LSODA')
        X1 = sol1.y
        pp_mr[x] = X1[0,0] 
        #pz_mr[x] = ppz_val(pp_mr[x],mu1,m_mr[0,x],D,R[x],phim[x])
        pp_mrI = X1[0,0]
        # except:
        #     print(x)
        #     print(len(pp_ml))
        #     print(len(X))
               
        #pp_mr[x] = sol.y 
        #print(pp_mr[x])

        #if (x<(num_rp-1)):
        arg2 = (mu1,m_pl[0,x],D,R[x],phim_l[x])
        sol2 = solve_ivp(Ppz, [z[x-1], z[x]], [pp_plI], t_eval=[z[x]],args = arg2, rtol = 1e-7,atol = 1e-9,method = 'LSODA')
        X2 = sol2.y
        pp_pl[x] = X2[0,0]             
        pp_plI = X2[0,0]

        arg3 = (mu1,m_pr[0,x],D,R[x],phim_r[x])
        sol3 = solve_ivp(Ppz, [z[x-1], z[x]], [pp_prI], t_eval=[z[x]],args = arg3, rtol = 1e-7,atol = 1e-9,method = 'LSODA')
        X3 = sol3.y
        pp_pr[x] = X3[0,0] 
        #pz_pr[x] = ppz_val(pp_pr[x],mu1,m_pr[0,x],D,R[x],phim_r[x])
        pp_prI = X3[0,0]

        #if (x<(num_rp-1)):
        arg4 = (mu1,m_rl[0,x],D,R_l[x],phim[x])
        sol4 = solve_ivp(Ppz, [z[x-1], z[x]], [pp_rlI], t_eval=[z[x]],args = arg4, rtol = 1e-7,atol = 1e-9,method = 'LSODA')#,max_step=1e-3)
        X4 = sol4.y
        pp_rl[x] = X4[0,0] 
        pp_rlI = X4[0,0]

        arg5 = (mu1,m_rr[0,x],D,R_r[x],phim[x])
        sol5 = solve_ivp(Ppz, [z[x-1], z[x]], [pp_rrI], t_eval=[z[x]],args = arg5, rtol = 1e-7,atol = 1e-9,method = 'LSODA')#,max_step=1e-3)
        X5 = sol5.y
        pp_rr[x] = X5[0,0] 
        pp_rrI = X5[0,0]
  
        #raise Exception('ODE solved')

    # pz_ml[1:-1] = (pp_ml[1:-1] - pp_ml[0:-2])/delx
    # pz_mr[1:-1] = (pp_mr[1:-1] - pp_mr[0:-2])/delx
    # pz_pl[1:-1] = (pp_pl[1:-1] - pp_pl[0:-2])/delx
    # pz_pr[1:-1] = (pp_pr[1:-1] - pp_pr[0:-2])/delx
    # pz_rl[1:-1] = (pp_rl[1:-1] - pp_rl[0:-2])/delx
    # pz_rr[1:-1] = (pp_rr[1:-1] - pp_rr[0:-2])/delx

    # pp_mr[num_rp-1] = pp_mr[num_rp-2]
    # pp_pr[num_rp-1] = pp_pr[num_rp-2]
    # pp_rr[num_rp-1] = pp_rr[num_rp-2]

    #pp_mr[1:-1] = pp_ml[2:]
    # d2 = 5
    # pp0 = 2
    # c2=0.3
   

    # pp = pp0 + c2*np.exp(-((zz-.6)*d2)**2) 

    # pp_ml = pp
    # pp_mr = pp
    # pp_pl = pp
    # pp_pr = pp
    # pp_rl = pp
    # pp_rr = pp

    pz_ml = ppz_val(pp_ml,mu1,m_ml[0,:],D,R,phim)
    pz_mr = ppz_val(pp_mr,mu1,m_mr[0,:],D,R,phim)
    pz_pl = ppz_val(pp_pl,mu1,m_pl[0,:],D,R,phim_l)
    pz_pr = ppz_val(pp_pr,mu1,m_pr[0,:],D,R,phim_r)
    pz_rl = ppz_val(pp_rl,mu1,m_rl[0,:],D,R_l,phim)
    pz_rr = ppz_val(pp_rr,mu1,m_rr[0,:],D,R_r,phim)
    # pz_ml = 0*ppz_val(pp_ml,mu1,m_ml[0,:],D,R,phim)
    # pz_mr = 0*ppz_val(pp_mr,mu1,m_mr[0,:],D,R,phim)
    # pz_pl = 0*ppz_val(pp_pl,mu1,m_pl[0,:],D,R,phim_l)
    # pz_pr = 0*ppz_val(pp_pr,mu1,m_pr[0,:],D,R,phim_r)
    # pz_rl = 0*ppz_val(pp_rl,mu1,m_rl[0,:],D,R_l,phim)
    # pz_rr = 0*ppz_val(pp_rr,mu1,m_rr[0,:],D,R_r,phim)

    # print(pz_ml[0])
    # print(pz_ml[1])
    # print(pz_ml[2])
    # print(pz_ml[3])
    # print(pz_ml[4])
    # print(pz_ml[5])
    #print(pz_mr[0])
    #print(pz_mr[1])

    #pz_ml = np.where(abs(pz_ml)<1e-10,0,pz_ml)
    #pz_mr = np.where(abs(pz_mr)<1e-10,0,pz_mr)


    #pz_pl2[0] = 1
#     pyplot.figure(10)
#     pyplot.plot(z,pz_pl)
#     pyplot.plot(z,pz_pr,'b',linestyle='dashed')
#     # pyplot.savefig('plot_pppz_rl.png',dpi = 300)
#     #pyplot.plot(z,pp_pr,'b',linestyle='dashed')
#     pyplot.savefig('plot_pppz_rl.png',dpi = 300)
#     pyplot.figure(40)
#     #pyplot.plot(z,phim_l)
#     pyplot.plot(z,m_pl[0,:])
#     pyplot.plot(z,m_pr[0,:],'b',linestyle='dashed')
#     pyplot.savefig('plot_mpl_rl.png',dpi = 300)
#     pyplot.figure(11)
#     pyplot.plot(z,pz_ml)
#     pyplot.plot(z,pz_mr,'b',linestyle='dashed')
#     #pyplot.plot(z,pp_mr,'b',linestyle='dashed')
#     pyplot.savefig('plot_ppmz_rl.png',dpi = 300)
#     pyplot.figure(41)
#     pyplot.plot(z,m_ml[0,:])
#     pyplot.plot(z,m_mr[0,:],'b',linestyle='dashed')
#     pyplot.savefig('plot_mml_rl.png',dpi = 300)
#    # pyplot.figure(30)
#     #pyplot.plot(z,pz_ml2)
#     #pyplot.ylim(0.49, 0.51) 
#     #pyplot.plot(z,pz_pl,'b',linestyle='dashed')
#     #pyplot.savefig('plot_pzml.png',dpi = 300)
#     pyplot.figure(20)
#     pyplot.plot(z,pp_pl)
#     pyplot.plot(z,pp_pr,'b',linestyle='dashed')
#     pyplot.savefig('plot_ppp_rl.png',dpi = 300)
    #pyplot.figure(30)
    #pyplot.plot(z,pp_rl)
    #pyplot.plot(z,pp_rr,'b',linestyle='dashed')
    #pyplot.savefig('plot_ppr_rl.png',dpi = 300)
    #print('ppz plotted')
    #raise Exception('ppz plotted')
    
    def Flux(pm,mu1,muw,D,R,pz):
        #f2 = np.empty( (num_eqn, num_rp) )
        #f4 = np.empty( (num_eqn, num_rp) )
    
        #for i in range(num_rp):
    
            # f2[0,i] = F2(mu1,muw[0,i])
            # f4[0,i] = F4(mu1,muw[0,i],pm[i])

        #print(f2)

        f2 = F2(mu1,muw)
        f3 = F3(mu1,muw)
        f4 = F4(mu1,muw,pm)            
        e = 1
        #e = 0
        #Flux = pm*f2/(f3 + D*f4/(R**2))
        Flux = (pm*f2/(f3 + D*f4/(R**2)))*(1 - D*f4*(R**2)*e*pz)
        #Flux = 2*pm*(R**3)*muw*f2*pp


        return(Flux)
        
        

    
    
    wave[0,0,:] = q_r - q_l            #Wave mu_w 
    wave[0,1,:] = phim_r - phim_l      #Wave phim
    wave[0,2,:] = R_r - R_l            #Wave R
    
    
    ## Calculate fluxes on both sides of the cell edge
    
    #mu_w varying across cell edge and R, phim const = avg
    Flux_muw_R = Flux(phim,mu1,m_mr,D,R,pz_mr)
    Flux_muw_L = Flux(phim,mu1,m_ml,D,R,pz_ml)
    # pyplot.figure(1000)
    # pyplot.plot(z,Flux_muw_R[0,:])
    # pyplot.plot(z,Flux_muw_L[0,:],'b',linestyle='dashed')
    # pyplot.savefig('plot_Flux_mrl.png',dpi = 400)
    #print(Flux_muw_R)
    
    #phim varying across cell edge and R, mu_w const = avg
    Flux_phim_R = Flux(phim_r,mu1,m_pr,D,R,pz_pr)
    Flux_phim_L = Flux(phim_l,mu1,m_pl,D,R,pz_pl)
    
    #raise Exception("MuI_Test")

    #R varying across cell edge and phim, mu_w const = avg
    Flux_R_R = Flux(phim,mu1,m_rr,D,R_r, pz_rr)
    Flux_R_L = Flux(phim,mu1,m_rl,D,R_l, pz_rl)
    
    # z = np.linspace(z0 - delx,z1 + delx,num_rp)
    # z1 = (z[0:-2] + z[1:-1])/2 
    # pyplot.figure()
    # pyplot.plot(z1,Flux_R_R[0,0:-2])
    # pyplot.plot(z1,Flux_R_L[0,1:-1],'b',linestyle='dashed')
    # pyplot.savefig('plot2_RF.png',dpi = 400)
    #pyplot.figure()
    #pyplot.plot(z, pp_mr)
    #pyplot.plot(z,m_mr[0,:],'b')
    #pyplot.savefig('plot2.png')
    # raise Exception('ODE solved')

    # if phi>=phim then Flux F = 0 since it is clogged
    Fm_r = np.where(m_mr>mu1,Flux_muw_R,0)
    Fm_l = np.where(m_ml>mu1,Flux_muw_L,0)

    Fp_r = np.where(m_pr>mu1,Flux_phim_R,0)
    Fp_l = np.where(m_pl>mu1,Flux_phim_L,0)
    
    Fr_r = np.where(m_rr>mu1,Flux_R_R,0)
    Fr_l = np.where(m_rl>mu1,Flux_R_L,0)

    # z = np.linspace(z0,z01,num_rp)
    #z1 = (z[0:-2] + z[1:-1])/2 
    # pyplot.figure(12)
    # pyplot.plot(z,Fp_r[0,:] - Fp_l[0,:])
    # #pyplot.plot(z1,,'b',linestyle='dashed')
    # pyplot.savefig('plot1_Fpdiff_rl.png',dpi = 300)
    # pyplot.figure(22)
    # pyplot.plot(z,Fm_r[0,:])
    # #pyplot.plot(z1,,'b',linestyle='dashed')
    # pyplot.savefig('plot_Fm_rl.png',dpi = 300)
    # pyplot.figure(13)
    # pyplot.plot(z,Fm_r[0,:] -Fm_l[0,:] )
    # #pyplot.plot(z1,Fm_l[0,1:-1],'b',linestyle='dashed')
    # pyplot.savefig('plot1_Fmdiff_rl.png',dpi = 300)
    # pyplot.figure(14)
    # pyplot.plot(z,Fr_r[0,:] - Fr_l[0,:])
    # #pyplot.plot(z1,Fr_l[0,1:-1],'b',linestyle='dashed')
    # pyplot.savefig('plot1_Frdiff_rl.png',dpi = 300)


    
    ##Calculate the speeds of each wave - partial dFlux/d*var* 
    
    
    x=0

    for x in range(num_rp):
        if(abs(q_r[0,x]-q_l[0,x])>1e-10):
            s[0,x] =(Fm_r[0,x] - Fm_l[0,x])/(q_r[0,x] - q_l[0,x])
        else:
            s[0,x] =0

        if(abs(phim_r[x]-phim_l[x])>1e-10):
            s[1,x] =(Fp_r[0,x] - Fp_l[0,x])/(phim_r[x] - phim_l[x])
        else:
            s[1,x] =0

        if(abs(R_r[x]-R_l[x])>1e-10):
            s[2,x] =(Fr_r[0,x] - Fr_l[0,x])/(R_r[x] - R_l[x])
        else:
            s[2,x] =0

    # z = np.linspace(z0 - delx,z01 + delx,num_rp)
    # z1 = (z[0:-2] + z[1:-1])/2 
    # pyplot.figure(300)
    # pyplot.plot(z1,(s[0,1:-1]))
    # #pyplot.plot(z1,(phim_r[1:-1] - phim_l[1:-1]))
    # pyplot.savefig('plot_s0.png',dpi = 300)
    # pyplot.figure(301)
    # #print((z1.shape[0]))
    # #print((z1.shape[1]))
    # pyplot.plot(z1,(s[1,1:-1]))
    # pyplot.savefig('plot1_s1.png',dpi = 300)
    # pyplot.figure(302)
    # pyplot.plot(z1,(s[2,1:-1]))
    # #pyplot.plot(z1,(phim_r[1:-1] - phim_l[1:-1]))
    # #pyplot.plot(z1,(phim[1:-1]))
    # # pyplot.plot(z1,s[0,1:-1],'b',linestyle='dashed')
    # pyplot.savefig('plot1_s2.png',dpi = 300)
    
    #print(s[0,2000:2050])
    #raise Exception('Speed phi')

    # s[0,:] = np.where(abs(q_r-q_l)>1e-10,(Fm_r - Fm_l)/(q_r - q_l),0)
    
    
    # s[1,:] = np.where(abs(phim_r-phim_l)>1e-10,(Fp_r - Fp_l)/(phim_r - phim_l),0)
    
    
    # s[2,:] = np.where(abs(R_r-R_l)>1e-10,(Fr_r - Fr_l)/(R_r - R_l),0)
           
    

# define the A del Q left and right going fluctuations 
# which means just seperating the +ve and -ve velocities
  
    s_index = np.zeros((2,num_rp))
    for mw in range(num_waves):
      s_index[0,:] = s[mw,:]
      amdq[0,:] += np.min(s_index,axis=0) * wave[0,mw,:]  
      apdq[0,:] += np.max(s_index,axis=0) * wave[0,mw,:]
      
    #   amdq[0,:] += np.min(s_index,axis=0) * wave[0,mw,:]  
    #   apdq[0,:] += np.max(s_index,axis=0) * wave[0,mw,:]
    #   pyplot.figure(str(mw))
    #   pyplot.plot(z,amdq[0,:])
    #   pyplot.plot(z,apdq[0,:])
    #   pyplot.savefig('plot_amdq'+str(mw)+'.png' ,dpi = 300)



    
    #Compute entropy fix
    if problem_data['efix']:
        transonic = (q_l[0,:] < 0.0) * (q_r[0,:] > 0.0)
        amdq[0,transonic] = -0.5 * q_l[0,transonic]**2
        apdq[0,transonic] = 0.5 * q_r[0,transonic]**2

    
    # pyplot.figure(303)
    # #print((z1.shape[0]))
    # #print((z1.shape[1]))
    # pyplot.plot(z1,(amdq[0,1:-1]))
    # pyplot.savefig('plot_amdq.png',dpi = 300)
    # pyplot.figure(304)
    # pyplot.plot(z1,(apdq[0,1:-1]))
    # pyplot.savefig('plot_apdq.png',dpi = 300)

    return wave, s, amdq, apdq
