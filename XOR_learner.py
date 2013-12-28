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
    layersize = 60
    V_L = -74*mV
    g_L = 25*nS
    C = 500*pF
    V_t = -54*mV
    V_r = -60*mV
    
    inp1 = [Poisson_Spiker(40.0) for x in range(layersize/2)]
    inp2 = [Poisson_Spiker(40.0) for x in range(layersize/2)]

    

    hidden_layer = [Neuron(C,g_L,V_L,normalvariate(425,200)*pA,V_t,V_r) for x in range(layersize)]

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
            S = Hedonistic_Synapse(tau_e = 20*ms,tau_g = 5*ms,W = W_val,V_rev = V_rev_val)
            synapses.append(S)
            neuron.inputs.append(S)
            inp.outputs.append(S)
        assert(len(inp.outputs)==layersize)


    for inp in inp2:
        r = randint(0,1)
        V_rev_val = r*(-70)*mV
        if(r == 0):
            W_val = expovariate(1.0/(2.4*nS))
        else:
            W_val = expovariate(1.0/(45*nS))
        for neuron in hidden_layer:
            S = Hedonistic_Synapse(tau_e = 20*ms,tau_g=5*ms,W = W_val,V_rev = V_rev_val)
            synapses.append(S)
            neuron.inputs.append(S)
            inp.outputs.append(S)

    
    for neuron in hidden_layer:
        assert(len(neuron.inputs)==layersize)
        r = randint(0,1)
        V_rev_val = r*(-70)*mV
        if(r == 0):
            W_val = expovariate(1.0/(2.4*nS))
        else:
            W_val = expovariate(1.0/(45*nS))
        S = Hedonistic_Synapse(tau_e = 20*ms,tau_g = 5*ms,W = W_val,V_rev = V_rev_val)
        synapses.append(S)
        neuron.outputs.append(S)
        outputneuron.inputs.append(S)
        
    
    return inp1,inp2,hidden_layer,outputneuron,synapses



def testsignal(x,y,inp1,inp2,hidden_layer,outputneuron,synapses,deltaT,trials):
    correct = (x+y)%2

    for inp in inp1:
        inp.setstate(x)
    for inp in inp2:
        inp.setstate(y)

        
    intrace = []
    outtrace = []
    inspikes = []
    outspikes = []
    for i in range(trials):
        trace = []
        gtrace = []
        for inp in inp1:
            trace.append(inp.generate(deltaT))
            gtrace += [x.G for x in inp.outputs]
        for inp in inp2:
            trace.append(inp.generate(deltaT))
            gtrace += [x.G for x in inp.outputs]

        intrace.append(gtrace)
        inspikes.append(trace)
        for neuron in hidden_layer:
            neuron.Update(deltaT)
        O = outputneuron.Update(deltaT)
        outspikes.append(O)
        outtrace.append(outputneuron.V)


        h = (2.0*correct-1.0)*(2*O-1)
        h = (2.0*correct - 1.0)*O
        for synapse in synapses:
            synapse.Update(h,0.3)

    return intrace,outtrace,inspikes,outspikes



def trainsignal(x,y,inp1,inp2,hidden_layer,outputneuron,synapses,deltaT,trials):
    correct = (x+y)%2

    for inp in inp1:
        inp.setstate(x)
    for inp in inp2:
        inp.setstate(y)

        

    for i in range(trials):
        for inp in inp1:
            inp.generate(deltaT)
        for inp in inp2:
            inp.generate(deltaT)

        for neuron in hidden_layer:
            neuron.Update(deltaT)
        O = outputneuron.Update(deltaT)

        
        h = (2.0*correct-1.0)*(2*O-1)
        h = (2.0*correct -1.0)*O
        for synapse in synapses:
            synapse.Update(h,0.3)



    
def train(inp1,inp2,hidden_layer,outputneuron,synapses,deltaT,trials,epochs):

    for e in range(epochs):
        intrace = []
        outtrace = []
        outspikes = []
        inspikes = []
        for x in [0,1]:
            for y in [0,1]:
                int,outt,ins,outs = testsignal(x,y,inp1,inp2,hidden_layer,outputneuron,synapses,deltaT,trials)
                intrace += int
                outtrace += outt
                inspikes += ins
                outspikes += outs
                print "\ttrained "+str(x)+str(y)
        fp = open("progressfile","w")
        dump([intrace,outtrace,inspikes,outspikes,[synapse.q for synapse in outputneuron.inputs]],fp)
        fp.close()
        print "finished epoch ",e

def test(inp1,inp2,hidden_layer,outputneuron,synapses,deltaT,trials):
    intrace = []
    outtrace = []
    inspikes = []
    outspikes = []
    for x in [0,1]:
        for y in [0,1]:
            int,outt,ins,outs = testsignal(x,y,inp1,inp2,hidden_layer,outputneuron,synapses,deltaT,trials)
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
    train(*tuple(list(vars)+[1.0/2000.0,1000, 20]))

    intrace,outtrace,inspikes,outspikes,qvals = test(*tuple(list(vars)+[ 1.0/2000.0, 100]))

    fp = open("neurondump_v2","w")
    
    dump([intrace,outtrace,inspikes,outspikes,qvals],fp)
    fp.close()


if __name__ == "__main__":
    main()
