#!/bin/python

from hedonistic import *
from math import exp
from random import randint,normalvariate,expovariate,random
from pickle import dump
from sys import exit
mV = 0.001
pF = 10.0**(-12)
nF = 10.0**(-9)
nS = 10.0**(-9)
pA = 10.0**(-12)
ms = 0.001

def correctval(x,y):
    return x*y

def setup():
    layersize = 60
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

    randspread = 1.0
    offset = 0

    inp1 = [Poisson_Spiker(40.0) for x in range(layersize/2)]
    inp2 = [Poisson_Spiker(40.0) for x in range(layersize/2)]
    outputneuron = Neuron(C,g_L,V_L,normalvariate(425*pA,2*pA),V_t,V_r)


    synapses = []
    c = 0
    V_rev_val = -70*mV
    W_val = 45*nS
    S = Hedonistic_Synapse(tau_r = tau_r_val,initialQ = randspread*(random()-0.5)+offset,tau_e = tau_e_val,tau_g= tau_g_val,tau_c = tau_c_val,delta_c = delta_c_val,W = W_val,V_rev = V_rev_val)    
    for inp in inp1:#range(len(inp1)):
        r = c%2#randint(0,1)
        r = 0
        c+=1
        offset = 3-3*r
        V_rev_val = (r)*(-70)*mV
        Q = 200*(2*r-1)

        if(r == 1):
            W_val = expovariate(1.0/(45*nS))
        else:
            W_val = expovariate(1.0/(2.4*nS))

        S = Hedonistic_Synapse(tau_r = tau_r_val,initialQ = randspread*(random()-0.5)+offset,tau_e = tau_e_val,tau_g= tau_g_val,tau_c = tau_c_val,delta_c = delta_c_val,W = W_val,V_rev = V_rev_val)    
        synapses.append(S)
        outputneuron.inputs.append(S)
        inp.outputs.append(S)
        assert(len(inp.outputs)==1)

    
    for inp in inp2:
        r = c%2#randint(0,1)
        r = 0
        c+=1
        Q = 200*(2*r-1)
        offset = 3-3*r
        V_rev_val = (r)*(-70)*mV
        offset = 3-3*r
        if(r == 1):
            W_val = expovariate(1.0/(45*nS))
        else:
            W_val = expovariate(1.0/(2.4*nS))

        S = Hedonistic_Synapse(tau_r=tau_r_val,initialQ = randspread*(random()-0.5)+offset, tau_e = tau_e_val,tau_g = tau_g_val,tau_c = tau_c_val,delta_c = delta_c_val,W = W_val,V_rev = V_rev_val)
        synapses.append(S)
        outputneuron.inputs.append(S)
        inp.outputs.append(S)


    return inp1,inp2,outputneuron,synapses



def testsignal(x,y,inp1,inp2,outputneuron,synapses,deltaT,trials):
    correct = correctval(x,y)#x(x*y)%2

    for inp in inp1:
        inp.setstate(x==1)
    for inp in inp2:
        inp.setstate(y==1)
        
    intrace = []
    inspikes = []
    outtrace = []
    outspikes = []
    c = 0
    print "testing: ",x,y
    print "correct: ",correct
    for i in range(trials):
        trace = []
        gtrace = []
        for inp in inp1:
            trace.append(inp.generate(deltaT))
            gtrace.append(inp.outputs[0].G)
            #print inp.outputs
            assert(len(inp.outputs) == 1)
        for inp in inp2:
            trace.append(inp.generate(deltaT))
            gtrace.append(inp.outputs[0].G)
            assert(len(inp.outputs) == 1)

#        intrace.append([synapse.G for synapse in synapses])
        intrace.append(gtrace)
        inspikes.append(trace)
        #print reduce(lambda x,y:x+y,[synapse.G for synapse in outputneuron.inputs])

        V_L = -74*mV
        g_L = 25*nS
        C = 500*pF
        V_t = -54*mV
        V_r = -60*mV
        I_tonic = outputneuron.I_tonic

        #A = (g_L+reduce(lambda x,y:x+y,[synapse.G for synapse in outputneuron.inputs]))/C
        #N = (-g_L*(-V_L)+I_tonic - reduce(lambda x,y:x+y,[synapse.G*(-synapse.V_rev) for synapse in outputneuron.inputs]))/C

        #print "I_tonic: ",I_tonic
        #2#Bprint "equilibrium: ",N/A
        #print "V: ",exp(-A*deltaT)*outputneuron.V+ (1.0/A) * (1-exp(-A*deltaT))* N

        #A = (g_L)/C
        #N = (-g_L*(-V_L)+I_tonic)/C        
        #print "trueV: ",exp(-A*deltaT)*outputneuron.V+ (1.0/A) * (1-exp(-A*deltaT))* N
        O = outputneuron.Update(deltaT)
        outtrace.append(outputneuron.V)
        outspikes.append(O)
        if O:

            O=1
        else:
            O=0
        h = (2*(correct)-1)*O#(2*O-1)
        if(h != 0 and random() > 0.5):
            print "outputspike: ",c
            c+=1
            print "h: ",h
        #if(h!=0):
        #    print [inp.outputs[0].q for inp in inp1]
        #for synapse in synapses:
        #    synapse.Update(h,0.3)
        #if(h!=0):
        #    print [inp.outputs[0].q for inp in inp1]
        #    exit()

    return intrace,outtrace,inspikes,outspikes



def trainsignal(x,y,inp1,inp2,outputneuron,synapses,deltaT,trials):
    correct = correctval(x,y)
    c=0
    total = 0
    print "trials: ",trials
    print "testing: ",x,y
    print "correct: ",correct
    for inp in inp1:
        inp.setstate(x)

    for inp in inp2:
        inp.setstate(y)

    for i in range(trials):
        for inp in inp1:
            inp.generate(deltaT)
        for inp in inp2:
            inp.generate(deltaT)


        O = outputneuron.Update(deltaT)
        if(O):
            O=1
        else:
            O=0#-0.01
            
        h = (2*(correct)-1)*O#(2*O-1)

        if(O >= 1):
            print "\t\toutputspike: ",c
            c+=1
            print "\t\th: ",h
#            print "\t\tresting voltage: "+str(outputneuron.V_L+(outputneuron.I_tonic)/outputneuron.g_L)+" I_tonic: "+str(outputneuron.I_tonic/pA)
            print "\t\tq0: "+str(outputneuron.inputs[0].q)+"\tq1: "+str(outputneuron.inputs[1].q)

    
        for synapse in synapses:
            synapse.Update(h,0.3)
            if(synapse.spiked):
                total+=1
    print "total spikes: ",total



    
def train(inp1,inp2,outputneuron,synapses,deltaT,trials,epochs):
    for e in range(epochs):
        for x in [0,1]:
            for y in [0,1]:
                trainsignal(x,y,inp1,inp2,outputneuron,synapses,deltaT,trials)

                for synapse in synapses:
                    synapse.trace_e = 0.0
                    synapse.G = 0.0

                print "\ttrained "+str(x)+str(y)
        print "finished epoch ",e

def test(inp1,inp2,outputneuron,synapses,deltaT,trials):
    intrace = []
    outtrace = []
    inspikes = []
    outspikes = []
    for x in [0,1]:
        for y in [0,1]:
            int,outt,ins,outs = testsignal(x,y,inp1,inp2,outputneuron,synapses,deltaT,trials)
            intrace += int
            outtrace += outt
            inspikes += ins
            outspikes += outs
    qvals = [synapse.q for synapse in outputneuron.inputs]
    return intrace,outtrace,inspikes,outspikes,qvals


def main():
    print "setting up!"
    vars = setup()
    print "setup finished, running!"
    train(*tuple(list(vars)+[1.0/2000.0,1000, 100]))

    intrace,outtrace,inspikes,outspikes,qvals = test(*tuple(list(vars)+[ 1.0/2000, 1000]))

    fp = open("testerdump_and","w")


    dump([intrace,outtrace,inspikes,outspikes,qvals],fp)
    fp.close()


if __name__ == "__main__":
    main()
