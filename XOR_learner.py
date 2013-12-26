#!/bin/python

from hedonistic import *
from random import randint,normalvariate,expovariate
from pickle import dump
mV = 0.001
pF = 10.0**(-12)
nS = 10.0**(-9)
pA = 10.0**(-12)
ms = 0.001



def setup():
    V_L = -74*mV
    g_L = 25*nS
    C = 500*pF
    V_t = -54*mV
    V_r = -60*mV
    
    inp1 = [Poisson_Spiker(40.0) for x in range(30)]
    inp2 = [Poisson_Spiker(40.0) for x in range(30)]

    

    hidden_layer = [Neuron(C,g_L,V_L,normalvariate(425,200)*pA,V_t,V_r) for x in range(60)]

    outputneuron = Neuron(C,g_L,V_L,normalvariate(425,200)*pA,V_t,V_r)

    synapses = []

    for inp in inp1:
        r = randint(0,1)
        V_rev_val = r*(-70)*mV
        if(r == 0):
            W_val = expovariate(1.0/(2.4*nS))
        else:
            W_val = expovariate(1.0/(45*nS))
        for neuron in hidden_layer:
            S = Hedonistic_Synapse(tau_e = 20*ms,W = W_val,V_rev = V_rev_val)
            synapses.append(S)
            neuron.inputs.append(S)
            inp.outputs.append(S)


    for inp in inp2:
        r = randint(0,1)
        V_rev_val = r*(-70)*mV
        if(r == 0):
            W_val = expovariate(1.0/(2.4*nS))
        else:
            W_val = expovariate(1.0/(45*nS))
        for neuron in hidden_layer:
            S = Hedonistic_Synapse(tau_e = 20*ms,W = W_val,V_rev = V_rev_val)
            synapses.append(S)
            neuron.inputs.append(S)
            inp.outputs.append(S)

    
    for neuron in hidden_layer:
        r = randint(0,1)
        V_rev_val = r*(-70)*mV
        if(r == 0):
            W_val = expovariate(1.0/(2.4*nS))
        else:
            W_val = expovariate(1.0/(45*nS))
        S = Hedonistic_Synapse(tau_e = 20*ms,W = W_val,V_rev = V_rev_val)
        neuron.outputs.append(S)
        outputneuron.inputs.append(S)
    
    return inp1,inp2,hidden_layer,outputneuron,synapses



def trainsignal(x,y,inp1,inp2,hidden_layer,outputneuron,synapses,deltaT,trials):
    correct = (x+y)%2

    for inp in inp1:
        inp.setstate(x)
    for inp in inp2:
        inp.setstate(y)

        
    intrace = []
    outtrace = []
    for i in range(trials):
        trace = []
        for inp in inp1:
            trace.append(inp.generate(deltaT))
        for inp in inp2:
            trace.append(inp.generate(deltaT))

        intrace.append(trace)

        for neuron in hidden_layer:
            neuron.Update(deltaT)
        O = outputneuron.Update(deltaT)
        outtrace.append(O)

        h = 0.0
        if(O):
            h = 2.0*correct-1.0
        for synapse in synapses:
            synapse.Update(h,0.3)

    return intrace,outtrace
    
def train(inp1,inp2,hidden_layer,outputneuron,synapses,deltaT,trials,epochs):
    for e in range(epochs):
        for x in [0,1]:
            for y in [0,1]:
                trainsignal(x,y,inp1,inp2,hidden_layer,outputneuron,synapses,deltaT,trials)
                print "\ttrained "+str(x)+str(y)
        print "finished epoch ",e

def test(inp1,inp2,hidden_layer,outputneuron,synapses,deltaT,trials):
    intrace = []
    outtrace = []
    for x in [0,1]:
        for y in [0,1]:
            int,outt = trainsignal(x,y,inp1,inp2,hidden_layer,outputneuron,synapses,deltaT,trials)
            intrace += int
            outtrace += outt
            
    return intrace,outtrace


def main():
    print "setting up!"
    vars = setup()
    print "setup finished, running!"
    train(*vars,deltaT = 1.0/80.0,trials = 40, epochs = 200)

    intrace,outtrace = test(*vars,deltaT = 1*ms,trials = 500)

    fp = open("neurondump","w")
    
    dump([intrace,outtrace],fp)
    fp.close()


if __name__ == "__main__":
    main()
