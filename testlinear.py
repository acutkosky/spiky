
from hedonistic import *
from random import random,expovariate
from math import exp,log
mV = 0.001
pF = 10.0**(-12)
nF = 10.0**(-9)
nS = 10.0**(-9)
pA = 10.0**(-12)
ms = 0.001


def run(l):



    V_L = -74*mV
    g_L = 25*nS
    C = 500*pF
    V_t = -54*mV
    V_r = -60*mV
    tau_r_val = 0.01*ms
    tau_e_val = 20*ms
    tau_g_val = 5*ms
    tau_c_val = 0.01*ms#500*ms
    delta_c_val = 0#1
    randspread = 0
    offset = 0
    Hz = 40

    spikers = []
    spiker = Poisson_Spiker(40)
    spiker.setstate(1)
    neuron = Neuron(C,g_L,V_L,425*pA,V_t,V_r)
    for i in range(0,l):
        spiker = Poisson_Spiker(Hz)
        spiker.setstate(1)
        V_rev_val = -0*mV
        W_val = 2.4*nS#expovariate(1.0/(2.4*nS))
        #if(i%2):
            #2BV_rev_val = -70*mV
            #W_val = 45*nS#expovariate(1.0/(45*nS))
        S = Hedonistic_Synapse(tau_r = tau_r_val,initialQ = randspread*(random()-0.5)+offset,tau_e = tau_e_val,tau_g= tau_g_val,tau_c = tau_c_val,delta_c = delta_c_val,W = W_val,V_rev = V_rev_val)
        spiker.outputs.append(S)
        neuron.inputs.append(S)
        spikers.append(spiker)
    deltaT = 0.5*ms
#    print "expected: ",spiker.rate*deltaT*2000
    c = 0
    s = 0
    av = 0
    for i in range(2000):

        for spiker in spikers:
            if(spiker.generate(deltaT)):
                
                s+=1
        if(neuron.Update(deltaT)):
            av += reduce(lambda x,y:x+y,[z.outputs[0].G for z in spikers[::2]])/len(spikers[::2])
            c+=1
 #   print "av: \t",av/c
  #  print "expected: \t",45*nS*tau_g_val*80*0.5
#    print "actual: ",s
    return c

def trueval(nume,numi):


    Hz = 40
    
    V_L = -74*mV
    g_L = 25*nS
    C = 500*pF
    V_t = -54*mV
    V_r = -60*mV
    tau_r_val = 0.01*ms
    tau_e_val = 20*ms
    tau_g = 5*ms
    tau_c_val = 0.01*ms#500*ms
    delta_c_val = 0#1
    randspread = 0
    offset = 0

    V_rev = -70*mV
    I_tonic = 500*pA
    W_e = 2.4*nS
    W_i = 45*nS
    G_e = W_e*Hz*0.5*tau_g
    G_i = W_i*Hz*0.5*tau_g


    T_e = W_e*0.5*tau_g*nume/C
    T_i = W_i*0.5*tau_g*nume/C
#    G_e = 5.0*10**(-10)
#    G_i = 9.6*10**(-9)
    
    L = -g_L/C-nume*G_e/C-numi*G_i/C
    N = g_L*V_L/C+numi*G_i*V_rev/C+I_tonic/C
#2    t = log((V_t+N/L)/(V_r+N/L))/L
    return -N/L#abs(N/L - (W_i*0.5*tau_g*num_i*V_rev/C)/(-(


print trueval(0,0)
#for i in range(60,62,2):
#2#B    print str(len(range(0,i,2)))+": "+str(reduce(lambda x,y:x+run(i),range(60),0)/60.0)
#    print str(len(range(0,i,2)))+": "+str(trueval(i,0))
