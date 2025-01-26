#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import math as math
import scipy.integrate as integrate
import scipy.special as special
#import random as random


# In[ ]:


# Units: cgs
st = 0.665245*(10**-24) # Thomson scattering cross section
c = 2.997925*(10**10) # speed of light
m = 9.10956*(10**-28) # mass of particles (electron)
q = 4.80325*(10**-10) # charge of particles (electron)


# In[ ]:


# Lorentz factor γ as a function of β = v/c and vice versa
def Lorentz(beta):
    return 1/(1+(beta**2))**0.5
def Beta(gamma):
    return (1-(1/gamma**2))**0.5
# Time evolution γ(t) for one particle [with A = (st*c*B**2)/(6*pi*m*c**2)]
def g_evolutionA(go,t,A):
    return go/(1+(A*go*t))
# Time evolution γ(t) for one particle [with B]
def g_evolutionB(go,t,B):
    AA = (st*c*(B**2))/(6*np.pi*m*(c**2))
    return go/(1+(AA*go*t))
# Characteristic time (t_syn: E = Eo/2)
def time_synA(go,A):
    return 1/(A*go)
def time_synB(go,B):
    AA = (st*c*(B**2))/(6*np.pi*m*(c**2))
    return 1/(AA*go)

# F(x): Synchrotron emission from a Single particle (x = v/v_c), integral with quad
def Fx_quad(x):
    return x*integrate.quad(lambda y: special.kv(5/3,y), x, np.inf)[0]
# F(x): with integrate.simpson [this will be used for the rest of this analysis]
def Fx_sim(xx):
    y2 = np.logspace(np.log10(xx),np.log10(xx+50),100) # it works faster/better with logspace , "xx+50" instead of "inf"
    res = [0]*len(y2)
    for i in range(len(y2)):
        res[i] = special.kv(5/3,y2[i])
    return xx*integrate.simpson(res, x=y2)
# F(x): Approximate equation, overestimates peak
def Fx_approx(x):
    return (4*math.pi/(math.sqrt(3)*math.gamma(1/3)))*((x/2)**(1/3))*math.exp(-x);
# F(x): with integrate.simpson, test the integral upper limit
def Fx_sim_test(xx,bb):
    y2 = np.logspace(np.log10(xx),np.log10(xx*bb),100) # upper limit: "xx*bb" instead of "inf"
    res = [0]*len(y2)
    for i in range(len(y2)):
        res[i] = special.kv(5/3,y2[i])
    return xx*integrate.simpson(res, x=y2)


# In[ ]:


# Instant injection of monoenergetic particles
def N_instant_mono(Q):
    return Q

# Continuous injection of monoenergetic particles
def N_constant_mono(g,Q,A):
    return (Q/A)*g**(-2)
# N_tot(t) - Continuous injection of monoenergetic particles
def Ntot_constant_mono(t,go,Q,A):
    return integrate.quad(lambda gg: N_constant_mono(gg,Q,A), g_evolutionA(go,t,A), go)[0]

# Instant powerlaw injection (γ^-p)
def N_instant_pl(g,t,Q,A,p):
    return Q*(g**(-p)) / ((1-(A*g*t))**(2-p))
# N_tot(t) - Instant powerlaw injection
def N_tot_instant_pl(t,gmin,gmax,Q,A,p):
    return integrate.quad(lambda g: N_instant_pl(g,t,Q,A,p), g_evolutionA(gmin,t,A), g_evolutionA(gmax,t,A))[0]

# Continuous powerlaw injection (γ^-p) - every case seperately
def N_constant_pl_i(g,t,Q,A,p,gmin,gmax): # t>tr , γmax(t) <γ< γmin.
    return (Q*(g**(-2))/(A*(1-p)))*((gmax**(1-p))-(gmin**(1-p)))
def N_constant_pl_ii(g,t,Q,A,p,gmin,gmax): # t<tr, γmin(t) <γ< γmin  &  t>tr, γmin(t) <γ< γmax(t).
    return (Q*(g**(-2))/(A*(1-p)))*(((g/(1-A*g*t))**(1-p))-(gmin**(1-p)))
def N_constant_pl_iii(g,t,Q,A,p,gmin,gmax): # t<tr, γmax(t) <γ< γmax  &  t>tr, γmin <γ< γmax.
    return (Q*(g**(-2))/(A*(1-p)))*((gmax**(1-p))-(g**(1-p)))
def N_constant_pl_iv(g,t,Q,A,p,gmin,gmax): # t<tr, γmin <γ< γmax(t).
    return (Q*(g**(-p-1))/(A*(1-p)))*(((1-A*g*t)**(p-1))-1)
# Continuous powerlaw injection (γ^-p) - general solution
def N_constant_pl(g,t,Q,A,p,gmin,gmax):
    tr = (gmax-gmin)/(A*gmax*gmin)
    if t<0:
        return # if input is t<0
    elif g<g_evolutionA(gmin,t,A):
        return # if γ outside the bounds
    elif g>gmax:
        return # if γ outside the bounds
    elif t<tr:
        if g<gmin:
            result = N_constant_pl_ii(g,t,Q,A,p,gmin,gmax)
        elif g<g_evolutionA(gmax,t,A):
            result = N_constant_pl_iv(g,t,Q,A,p,gmin,gmax)
        else:
            result = N_constant_pl_iii(g,t,Q,A,p,gmin,gmax)
    elif t==tr:
        if g<gmin:
            result = N_constant_pl_ii(g,t,Q,A,p,gmin,gmax)
        else:
            result = N_constant_pl_iii(g,t,Q,A,p,gmin,gmax)
    elif t>tr:
        if g<g_evolutionA(gmax,t,A):
            result = N_constant_pl_ii(g,t,Q,A,p,gmin,gmax)
        elif g<gmin:
            result = N_constant_pl_i(g,t,Q,A,p,gmin,gmax)
        else:
            result = N_constant_pl_iii(g,t,Q,A,p,gmin,gmax)
    return result

# N_tot(t) - Continuous powerlaw injection (t<t_r)
def Ntot_constant_pl_ltr(t,gmin,gmax,Q,A,p):
    return integrate.quad(lambda g: N_constant_pl_ii(g,t,Q,A,p,gmin,gmax), g_evolutionA(gmin,t,A), gmin)[0] + integrate.quad(lambda g: N_constant_pl_iv(g,t,Q,A,p,gmin,gmax), gmin, g_evolutionA(gmax,t,A))[0] + integrate.quad(lambda g: N_constant_pl_iii(g,t,Q,A,p,gmin,gmax), g_evolutionA(gmax,t,A), gmax)[0]
# N_tot(t) - Continuous powerlaw injection (t>t_r)
def Ntot_constant_pl_gtr(t,gmin,gmax,Q,A,p):
    return integrate.quad(lambda g: N_constant_pl_ii(g,t,Q,A,p,gmin,gmax), g_evolutionA(gmin,t,A), g_evolutionA(gmax,t,A))[0] + integrate.quad(lambda g: N_constant_pl_i(g,t,Q,A,p,gmin,gmax), g_evolutionA(gmax,t,A), gmin)[0] + integrate.quad(lambda g: N_constant_pl_iii(g,t,Q,A,p,gmin,gmax), gmin, gmax)[0]
# N_tot(t) - Continuous powerlaw injection (for every t)
def Ntot_constant_pl(t,gmin,gmax,Q,A,p):
    tr = (gmax-gmin)/(A*gmax*gmin)
    if t<0:
        return
    elif t<tr:
        return Ntot_constant_pl_ltr(t,gmin,gmax,Q,A,p)
    else: #t>=tr
        return Ntot_constant_pl_gtr(t,gmin,gmax,Q,A,p)


# In[ ]:


# 1a. Function for plot N(γ,t) - we calculate and plot each range of γ seperately
def Plot_N(t,Q,A,p,gmin,gmax,co):
    tsyn_gmin = time_synA(gmin,A) # t_syn(γ_min) = 1/(A*γ_min)    
    tr = (gmax-gmin)/(A*gmax*gmin)
    if t<0:
        return
    elif t<tr:
        gg1a = np.linspace(g_evolutionA(gmax,t,A), gmax, 60)
        plt.plot(gg1a,N_constant_pl_iii(gg1a,t,Q,A,p,gmin,gmax), label="t={:0.1e}" .format(t/tsyn_gmin) + r"$\cdot t_{syn}(γ_{min})$"+ "={:0.1e} s" .format(t), c=co)
        gg2a = np.linspace(gmin,g_evolutionA(gmax,t,A), 60)
        plt.plot(gg2a,N_constant_pl_iv(gg2a,t,Q,A,p,gmin,gmax), c=co)
        gg3a = np.linspace(g_evolutionA(gmin,t,A), gmin, 60)
        plt.plot(gg3a,N_constant_pl_ii(gg3a,t,Q,A,p,gmin,gmax), c=co)
    elif t>tr:
        gg1b = np.linspace(gmin, gmax, 60)
        plt.plot(gg1b,N_constant_pl_iii(gg1b,t,Q,A,p,gmin,gmax), label="t={:0.1e}" .format(t/tsyn_gmin) + r"$\cdot t_{syn}(γ_{min})$"+ "={:0.1e} s" .format(t), c=co)
        gg2b = np.linspace(g_evolutionA(gmax,t,A), gmin, 60)
        plt.plot(gg2b,N_constant_pl_i(gg2b,t,Q,A,p,gmin,gmax), c=co)
        gg3b = np.linspace(g_evolutionA(gmin,t,A), g_evolutionA(gmax,t,A), 60)
        plt.plot(gg3b,N_constant_pl_ii(gg3b,t,Q,A,p,gmin,gmax), c=co)
    elif t==tr:
        g1 = np.linspace(gmin, gmax, 60)
        plt.plot(g1,N_constant_pl_iii(g1,t,Q,A,p,gmin,gmax), label="$t_r$={:0.1e}" .format(tr/tsyn_gmin) + r"$\cdot t_{syn}(γ_{min})$", c='k')
        g3 = np.linspace(g_evolutionA(gmin,t,A), gmin, 60)
        plt.plot(g3,N_constant_pl_ii(g3,t,Q,A,p,gmin,gmax), c='k')
# 1b. Function for plot N(γ,t) - no labels
def Plot_N_noLabel(t,Q,A,p,gmin,gmax,co):
    tsyn_gmin = time_synA(gmin,A) # t_syn(γ_min) = 1/(A*γ_min)    
    tr = (gmax-gmin)/(A*gmax*gmin)
    if t<0:
        return
    elif t<tr:
        gg1a = np.linspace(g_evolutionA(gmax,t,A), gmax, 60)
        plt.plot(gg1a,N_constant_pl_iii(gg1a,t,Q,A,p,gmin,gmax), c=co)
        gg2a = np.linspace(gmin,g_evolutionA(gmax,t,A), 60)
        plt.plot(gg2a,N_constant_pl_iv(gg2a,t,Q,A,p,gmin,gmax), c=co)
        gg3a = np.linspace(g_evolutionA(gmin,t,A), gmin, 60)
        plt.plot(gg3a,N_constant_pl_ii(gg3a,t,Q,A,p,gmin,gmax), c=co)
    elif t>tr:
        gg1b = np.linspace(gmin, gmax, 60)
        plt.plot(gg1b,N_constant_pl_iii(gg1b,t,Q,A,p,gmin,gmax), c=co)
        gg2b = np.linspace(g_evolutionA(gmax,t,A), gmin, 60)
        plt.plot(gg2b,N_constant_pl_i(gg2b,t,Q,A,p,gmin,gmax), c=co)
        gg3b = np.linspace(g_evolutionA(gmin,t,A), g_evolutionA(gmax,t,A), 60)
        plt.plot(gg3b,N_constant_pl_ii(gg3b,t,Q,A,p,gmin,gmax), c=co)
    elif t==tr:
        g1 = np.linspace(gmin, gmax, 60)
        plt.plot(g1,N_constant_pl_iii(g1,t,Q,A,p,gmin,gmax), c='k')
        g3 = np.linspace(g_evolutionA(gmin,t,A), gmin, 60)
        plt.plot(g3,N_constant_pl_ii(g3,t,Q,A,p,gmin,gmax), c='k')
# 1c. Function for plot N(γ,t) - no labels & subplot
def Plot_N_noLabel_subplot(t,Q,A,p,gmin,gmax,co,axes,x):
    tsyn_gmin = time_synA(gmin,A) # t_syn(γ_min) = 1/(A*γ_min)    
    tr = (gmax-gmin)/(A*gmax*gmin)
    if t<0:
        return
    elif t<tr:
        gg1a = np.linspace(g_evolutionA(gmax,t,A), gmax, 60)
        axes[x].plot(gg1a,N_constant_pl_iii(gg1a,t,Q,A,p,gmin,gmax), c=co)
        gg2a = np.linspace(gmin,g_evolutionA(gmax,t,A), 60)
        axes[x].plot(gg2a,N_constant_pl_iv(gg2a,t,Q,A,p,gmin,gmax), c=co)
        gg3a = np.linspace(g_evolutionA(gmin,t,A), gmin, 60)
        axes[x].plot(gg3a,N_constant_pl_ii(gg3a,t,Q,A,p,gmin,gmax), c=co)
    elif t>tr:
        gg1b = np.linspace(gmin, gmax, 60)
        axes[x].plot(gg1b,N_constant_pl_iii(gg1b,t,Q,A,p,gmin,gmax), c=co)
        gg2b = np.linspace(g_evolutionA(gmax,t,A), gmin, 60)
        axes[x].plot(gg2b,N_constant_pl_i(gg2b,t,Q,A,p,gmin,gmax), c=co)
        gg3b = np.linspace(g_evolutionA(gmin,t,A), g_evolutionA(gmax,t,A), 60)
        axes[x].plot(gg3b,N_constant_pl_ii(gg3b,t,Q,A,p,gmin,gmax), c=co)
    elif t==tr:
        g1 = np.linspace(gmin, gmax, 60)
        axes[x].plot(g1,N_constant_pl_iii(g1,t,Q,A,p,gmin,gmax), c='k')
        g3 = np.linspace(g_evolutionA(gmin,t,A), gmin, 60)
        axes[x].plot(g3,N_constant_pl_ii(g3,t,Q,A,p,gmin,gmax), c='k')
        
# 2a. Function for plot (γ^p)*N(γ,t)
def Plot_Ngp(t,Q,A,p,gmin,gmax,co):
    tsyn_gmin = time_synA(gmin,A) # t_syn(γ_min) = 1/(A*γ_min)    
    tr = (gmax-gmin)/(A*gmax*gmin)
    if t<0:
        return
    elif t<tr:
        gg1a = np.linspace(g_evolutionA(gmax,t,A), gmax, 60)
        plt.plot(gg1a,(gg1a**p)*N_constant_pl_iii(gg1a,t,Q,A,p,gmin,gmax), label="t={:0.1e}" .format(t/tsyn_gmin) + r"$\cdot t_{syn}(γ_{min})$"+ "={:0.1e} s" .format(t), c=co)
        gg2a = np.linspace(gmin,g_evolutionA(gmax,t,A), 60)
        plt.plot(gg2a,(gg2a**p)*N_constant_pl_iv(gg2a,t,Q,A,p,gmin,gmax), c=co)
        gg3a = np.linspace(g_evolutionA(gmin,t,A), gmin, 60)
        plt.plot(gg3a,(gg3a**p)*N_constant_pl_ii(gg3a,t,Q,A,p,gmin,gmax), c=co)
    elif t>tr:
        gg1b = np.linspace(gmin, gmax, 60)
        plt.plot(gg1b,(gg1b**p)*N_constant_pl_iii(gg1b,t,Q,A,p,gmin,gmax), label="t={:0.1e}" .format(t/tsyn_gmin) + r"$\cdot t_{syn}(γ_{min})$"+ "={:0.1e} s" .format(t), c=co)
        gg2b = np.linspace(g_evolutionA(gmax,t,A), gmin, 60)
        plt.plot(gg2b,(gg2b**p)*N_constant_pl_i(gg2b,t,Q,A,p,gmin,gmax), c=co)
        gg3b = np.linspace(g_evolutionA(gmin,t,A), g_evolutionA(gmax,t,A), 60)
        plt.plot(gg3b,(gg3b**p)*N_constant_pl_ii(gg3b,t,Q,A,p,gmin,gmax), c=co)
    elif t==tr:
        g1 = np.linspace(gmin, gmax, 60)
        plt.plot(g1,(g1**p)*N_constant_pl_iii(g1,t,Q,A,p,gmin,gmax), label="$t_r$={:0.1e}" .format(tr/tsyn_gmin) + r"$\cdot t_{syn}(γ_{min})$", c='k')
        g3 = np.linspace(g_evolutionA(gmin,t,A), gmin, 60)
        plt.plot(g3,(g3**p)*N_constant_pl_ii(g3,t,Q,A,p,gmin,gmax), c='k')
# 2b. Function for plot (γ^p)*N(γ,t) - no labels
def Plot_Ngp_noLabel(t,Q,A,p,gmin,gmax,co):
    tsyn_gmin = time_synA(gmin,A) # t_syn(γ_min) = 1/(A*γ_min)    
    tr = (gmax-gmin)/(A*gmax*gmin)
    if t<0:
        return
    elif t<tr:
        gg1a = np.linspace(g_evolutionA(gmax,t,A), gmax, 60)
        plt.plot(gg1a,(gg1a**p)*N_constant_pl_iii(gg1a,t,Q,A,p,gmin,gmax), c=co)
        gg2a = np.linspace(gmin,g_evolutionA(gmax,t,A), 60)
        plt.plot(gg2a,(gg2a**p)*N_constant_pl_iv(gg2a,t,Q,A,p,gmin,gmax), c=co)
        gg3a = np.linspace(g_evolutionA(gmin,t,A), gmin, 60)
        plt.plot(gg3a,(gg3a**p)*N_constant_pl_ii(gg3a,t,Q,A,p,gmin,gmax), c=co)
    elif t>tr:
        gg1b = np.linspace(gmin, gmax, 60)
        plt.plot(gg1b,(gg1b**p)*N_constant_pl_iii(gg1b,t,Q,A,p,gmin,gmax), c=co)
        gg2b = np.linspace(g_evolutionA(gmax,t,A), gmin, 60)
        plt.plot(gg2b,(gg2b**p)*N_constant_pl_i(gg2b,t,Q,A,p,gmin,gmax), c=co)
        gg3b = np.linspace(g_evolutionA(gmin,t,A), g_evolutionA(gmax,t,A), 60)
        plt.plot(gg3b,(gg3b**p)*N_constant_pl_ii(gg3b,t,Q,A,p,gmin,gmax), c=co)
    elif t==tr:
        g1 = np.linspace(gmin, gmax, 60)
        plt.plot(g1,(g1**p)*N_constant_pl_iii(g1,t,Q,A,p,gmin,gmax), c='k')
        g3 = np.linspace(g_evolutionA(gmin,t,A), gmin, 60)
        plt.plot(g3,(g3**p)*N_constant_pl_ii(g3,t,Q,A,p,gmin,gmax), c='k')
# 2c. Function for plot (γ^p)*N(γ,t) - no labels & subplot
def Plot_Ngp_noLabel_subplot(t,Q,A,p,gmin,gmax,co,axes,x):
    tsyn_gmin = time_synA(gmin,A) # t_syn(γ_min) = 1/(A*γ_min)    
    tr = (gmax-gmin)/(A*gmax*gmin)
    if t<0:
        return
    elif t<tr:
        gg1a = np.linspace(g_evolutionA(gmax,t,A), gmax, 60)
        axes[x].plot(gg1a,(gg1a**p)*N_constant_pl_iii(gg1a,t,Q,A,p,gmin,gmax), c=co)
        gg2a = np.linspace(gmin,g_evolutionA(gmax,t,A), 60)
        axes[x].plot(gg2a,(gg2a**p)*N_constant_pl_iv(gg2a,t,Q,A,p,gmin,gmax), c=co)
        gg3a = np.linspace(g_evolutionA(gmin,t,A), gmin, 60)
        axes[x].plot(gg3a,(gg3a**p)*N_constant_pl_ii(gg3a,t,Q,A,p,gmin,gmax), c=co)
    elif t>tr:
        gg1b = np.linspace(gmin, gmax, 60)
        axes[x].plot(gg1b,(gg1b**p)*N_constant_pl_iii(gg1b,t,Q,A,p,gmin,gmax), c=co)
        gg2b = np.linspace(g_evolutionA(gmax,t,A), gmin, 60)
        axes[x].plot(gg2b,(gg2b**p)*N_constant_pl_i(gg2b,t,Q,A,p,gmin,gmax), c=co)
        gg3b = np.linspace(g_evolutionA(gmin,t,A), g_evolutionA(gmax,t,A), 60)
        axes[x].plot(gg3b,(gg3b**p)*N_constant_pl_ii(gg3b,t,Q,A,p,gmin,gmax), c=co)
    elif t==tr:
        g1 = np.linspace(gmin, gmax, 60)
        axes[x].plot(g1,(g1**p)*N_constant_pl_iii(g1,t,Q,A,p,gmin,gmax), c='k')
        g3 = np.linspace(g_evolutionA(gmin,t,A), gmin, 60)
        axes[x].plot(g3,(g3**p)*N_constant_pl_ii(g3,t,Q,A,p,gmin,gmax), c='k')


# In[ ]:


# "Fx_sim(xx)" is F(x) with integrate.simpson [this will be used for the rest of this analysis]

# F(v,γ): the "spectrum" from a single particle with γ
def Fx_g(v,g,B):
    w_B = (q*B)/(m*c) # ω_B gyro-frequency
    vc = (3*w_B/(4*np.pi))*(g**2) # characteristic frequency v_c(γ)
    return Fx_sim(v/vc)
# j_syn(v,γ): the spectrum from a single particle with γ / sin(a)=1
def j_syn(v,g,B):
    j_o = np.sqrt(3)*(q**3)*B/(m*(c**2)) # normalisation parameter = sqrt(3)*(q**2)*ω_B/c
    return j_o*Fx_g(v,g,B)

# J_syn(v,t) from continuous injection of monoenergetic particles (with g_star)
def Jsyn_constant_mono(v,t,Q,B,g_star):
    A = (st*(B**2))/(6*m*c*np.pi)
    gg = np.logspace(np.log10(g_evolutionB(g_star,t,B)), np.log10(g_star), 200) # range of γ, at time t
    y1 = [0]*len(gg)
    for i in range(len(gg)):
        y1[i] = N_constant_mono(gg[i],Q,A)*j_syn(v,gg[i],B) # integral with simpson
    return integrate.simpson(y1, x=gg)
    
# J_syn(v,t) from instant power-law injection (γ^-p)
def Jsyn_instant_pl(v,t,Q,B,p,gmin,gmax):
    A = (st*(B**2))/(6*m*c*np.pi)
    gg = np.logspace(np.log10(g_evolutionB(gmin,t,B)), np.log10(g_evolutionB(gmax,t,B)), 200) # range of γ, at time t
    y1 = [0]*len(gg)
    for i in range(len(gg)):
        y1[i] = N_instant_pl(gg[i],t,Q,A,p)*j_syn(v,gg[i],B) # integral with simpson
    return integrate.simpson(y1, x=gg) 

# J_syn(v,t) from continuous power-law injection (γ^-p)
def Jsyn_constant_pl_tr1(v,t,Q,B,p,gmin,gmax): # t<tr
    A = (st*(B**2))/(6*m*c*np.pi)
    g2 = np.logspace(np.log10(g_evolutionB(gmin,t,B)), np.log10(gmin), 150) # logspace is better for big ranges
    y2 = [0]*len(g2)
    g4 = np.logspace(np.log10(gmin), np.log10(g_evolutionB(gmax,t,B)), 150)
    y4 = [0]*len(g4)
    g3 = np.logspace(np.log10(g_evolutionB(gmax,t,B)), np.log10(gmax), 150)
    y3 = [0]*len(g3)
    for i in range(len(g2)):
        y2[i] = N_constant_pl_ii(g2[i],t,Q,A,p,gmin,gmax)*j_syn(v,g2[i],B)
        y4[i] = N_constant_pl_iv(g4[i],t,Q,A,p,gmin,gmax)*j_syn(v,g4[i],B)
        y3[i] = N_constant_pl_iii(g3[i],t,Q,A,p,gmin,gmax)*j_syn(v,g3[i],B)
    I = integrate.simpson(y2, x=g2) + integrate.simpson(y4, x=g4) + integrate.simpson(y3, x=g3)
    return I
def Jsyn_constant_pl_tr2(v,t,Q,B,p,gmin,gmax): # t>tr
    A = (st*(B**2))/(6*m*c*np.pi)
    g2 = np.logspace(np.log10(g_evolutionB(gmin,t,B)), np.log10(g_evolutionB(gmax,t,B)), 150)
    y2 = [0]*len(g2)
    g1 = np.logspace(np.log10(g_evolutionB(gmax,t,B)), np.log10(gmin), 150)
    y1 = [0]*len(g1)
    g3 = np.logspace(np.log10(gmin), np.log10(gmax), 150)
    y3 = [0]*len(g3)
    for i in range(len(g2)):
        y2[i] = N_constant_pl_ii(g2[i],t,Q,A,p,gmin,gmax)*j_syn(v,g2[i],B)
        y1[i] = N_constant_pl_i(g1[i],t,Q,A,p,gmin,gmax)*j_syn(v,g1[i],B)
        y3[i] = N_constant_pl_iii(g3[i],t,Q,A,p,gmin,gmax)*j_syn(v,g3[i],B)
    I = integrate.simpson(y2, x=g2) + integrate.simpson(y1, x=g1) + integrate.simpson(y3, x=g3)
    return I
# J_syn(v,t) in one function, with integrate.simpson
def Jsyn_constant_pl(v,t,Q,B,p,gmin,gmax):
    A = (st*(B**2))/(6*m*c*np.pi)
    tr = (gmax-gmin)/(A*gmax*gmin)
    if t<0:
        return # if input is t<0
    elif t==0:
        I = 0 # J_syn(t=0) = 0, no particles have been injected yet
    elif t<tr:
        g2 = np.logspace(np.log10(g_evolutionB(gmin,t,B)), np.log10(gmin), 150) # logspace is better for big ranges
        y2 = [0]*len(g2)
        g4 = np.logspace(np.log10(gmin), np.log10(g_evolutionB(gmax,t,B)), 150)
        y4 = [0]*len(g4)
        g3 = np.logspace(np.log10(g_evolutionB(gmax,t,B)), np.log10(gmax), 150)
        y3 = [0]*len(g3)
        for i in range(len(g2)):
            y2[i] = N_constant_pl_ii(g2[i],t,Q,A,p,gmin,gmax)*j_syn(v,g2[i],B)
            y4[i] = N_constant_pl_iv(g4[i],t,Q,A,p,gmin,gmax)*j_syn(v,g4[i],B)
            y3[i] = N_constant_pl_iii(g3[i],t,Q,A,p,gmin,gmax)*j_syn(v,g3[i],B)
        I = integrate.simpson(y2, x=g2) + integrate.simpson(y4, x=g4) + integrate.simpson(y3, x=g3)
    elif t==tr:
        g2 = np.logspace(np.log10(g_evolutionB(gmin,t,B)), np.log10(gmin), 150)
        y2 = [0]*len(g2)
        g3 = np.logspace(np.log10(gmin), np.log10(gmax), 150)
        y3 = [0]*len(g3)
        for i in range(len(g2)):
            y2[i] = N_constant_pl_ii(g2[i],t,Q,A,p,gmin,gmax)*j_syn(v,g2[i],B)
            y3[i] = N_constant_pl_iii(g3[i],t,Q,A,p,gmin,gmax)*j_syn(v,g3[i],B)
        I = integrate.simpson(y2, x=g2) + integrate.simpson(y3, x=g3)
    else: # t>tr
        g2 = np.logspace(np.log10(g_evolutionB(gmin,t,B)), np.log10(g_evolutionB(gmax,t,B)), 150)
        y2 = [0]*len(g2)
        g1 = np.logspace(np.log10(g_evolutionB(gmax,t,B)), np.log10(gmin), 150)
        y1 = [0]*len(g1)
        g3 = np.logspace(np.log10(gmin), np.log10(gmax), 150)
        y3 = [0]*len(g3)
        for i in range(len(g2)):
            y2[i] = N_constant_pl_ii(g2[i],t,Q,A,p,gmin,gmax)*j_syn(v,g2[i],B)
            y1[i] = N_constant_pl_i(g1[i],t,Q,A,p,gmin,gmax)*j_syn(v,g1[i],B)
            y3[i] = N_constant_pl_iii(g3[i],t,Q,A,p,gmin,gmax)*j_syn(v,g3[i],B)
        I = integrate.simpson(y2, x=g2) + integrate.simpson(y1, x=g1) + integrate.simpson(y3, x=g3)
    return I


# In[ ]:


# L(t) calculation with: integral.simpson + log(v), a fast method
def Luminosity_instant_pl(t,vmin,vmax,Q,B,p,gmin,gmax):
    A = (st*(B**2))/(6*m*c*np.pi)
    #vnew_s = np.linspace(np.log10(vmin), np.log10(vmax), 200) # linspace (slower)
    vnew_s = np.logspace(np.log10(np.log10(vmin)), np.log10(np.log10(vmax)), 200) # same range, with logspace (faster)
    # Parameter change for the integral: vnew_s is a new parameter, vnew_s = np.log10(v) 
    y1 = [0]*len(vnew_s)
    for i in range(len(vnew_s)):
        y1[i] = (Jsyn_instant_pl(10**vnew_s[i],t,Q,B,p,gmin,gmax))*(10**vnew_s[i])*np.log(10)
    return integrate.simpson(y1, x=vnew_s)
def Luminosity_constant_pl(t,vmin,vmax,Q,B,p,gmin,gmax):
    A = (st*(B**2))/(6*m*c*np.pi)
    vnew_s = np.logspace(np.log10(np.log10(vmin)), np.log10(np.log10(vmax)), 100) # same range, with logspace (faster)
    # Parameter change for the integral: vnew_s is a new parameter, vnew_s = np.log10(v) 
    y1 = [0]*len(vnew_s)
    for i in range(len(vnew_s)):
        y1[i] = (Jsyn_constant_pl(10**vnew_s[i],t,Q,B,p,gmin,gmax))*(10**vnew_s[i])*np.log(10)
    return integrate.simpson(y1, x=vnew_s)

# luminosity, with the Doppler shift (Observer's frame)
def Luminosity_constant_pl_Dop(t,vmin,vmax,D,Q,B,p,gmin,gmax): # input v = v_rest = v_obs/D
    A = (st*(B**2))/(6*m*c*np.pi)
    vnew_s = np.logspace(np.log10(np.log10(vmin)), np.log10(np.log10(vmax)), 100) # logspace (faster)
    y1 = [0]*len(vnew_s)
    for i in range(len(vnew_s)):
        y1[i] = (D**3)*(Jsyn_constant_pl(10**vnew_s[i],t,Q,B,p,gmin,gmax))*(10**vnew_s[i])*np.log(10)
    return integrate.simpson(y1, x=vnew_s) # Input: t=t_rest, afterwards we must: t_obs = t/D for plot

# Flux = L/4πr^2
def Flux(L,r):
    rcm = r*3.08567758*10**24 # cm (r in Mpc)
    F = L/(4*math.pi*(rcm)**2) # erg/s/cm^2
    return F

# Doppler factor
def D(g,u):
    b = np.sqrt(1-(1/g**2))
    return 1/(g*(1-b*np.cos(u)))

