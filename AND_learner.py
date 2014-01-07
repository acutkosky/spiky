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
    tau_r_val = 0.01*ms
    Q = 0
    randspread = 3.0
    offset = 0.0
    
    
    inp1 = [Poisson_Spiker(40.0) for x in range(layersize/2)]
    inp2 = [Poisson_Spiker(40.0) for x in range(layersize/2)]

    

    hidden_layer = [Neuron(C,g_L,V_L,normalvariate(425,200)*pA,V_t,V_r) for x in range(layersize)]

    outputneuron = Neuron(C,g_L,V_L,normalvariate(425,200)*pA,V_t,V_r)

    synapses = []

    for inp in inp1:
        r = randint(0,1)
#        r = 0
#        r = randint(0,1)*randint(0,1)
        V_rev_val = r*(-70)*mV
        if(r == 0):
            W_val = expovariate(1.0/(2.4*nS))
        else:
            W_val = expovariate(1.0/(45*nS))
        for neuron in hidden_layer:
            Q = randspread*(random()-0.5)+offset
            S = Hedonistic_Synapse(initialQ = Q,tau_r = tau_r_val,tau_e = 20*ms,tau_g = 5*ms,W = W_val,V_rev = V_rev_val)
            synapses.append(S)
            neuron.inputs.append(S)
            inp.outputs.append(S)
        assert(len(inp.outputs)==layersize)


    for inp in inp2:
        r = randint(0,1)
#        r = 0
#        r = randint(0,1)*randint(0,1)
        V_rev_val = r*(-70)*mV
        if(r == 0):
            W_val = expovariate(1.0/(2.4*nS))
        else:
            W_val = expovariate(1.0/(45*nS))
        for neuron in hidden_layer:
            Q = randspread*(random()-0.5)+offset
                    
            S = Hedonistic_Synapse(initialQ = Q,tau_e = 20*ms,tau_r=tau_r_val,tau_g=5*ms,W = W_val,V_rev = V_rev_val)
            synapses.append(S)
            neuron.inputs.append(S)
            inp.outputs.append(S)

    
    for neuron in hidden_layer:
        assert(len(neuron.inputs)==layersize)
        r = randint(0,1)
#        r = randint(0,1)*randint(0,1)
#        r = 0
        V_rev_val = r*(-70)*mV
        if(r == 0):
            W_val = expovariate(1.0/(2.4*nS))
        else:
            W_val = expovariate(1.0/(45*nS))
        Q = randspread*(random()-0.5)+offset
        S = Hedonistic_Synapse(initialQ = Q, tau_e = 20*ms,tau_g = 5*ms,tau_r=tau_r_val,W = W_val,V_rev = V_rev_val)
        synapses.append(S)
        neuron.outputs.append(S)
        outputneuron.inputs.append(S)
        
    
    return inp1,inp2,hidden_layer,outputneuron,synapses



def testsignal(x,y,inp1,inp2,hidden_layer,outputneuron,synapses,deltaT,trials):

    correct = (x*y)%2
    print "correct: ",correct
    for inp in inp1:
        inp.setstate(x)
    for inp in inp2:
        inp.setstate(y)


        
    intrace = []
    outtrace = []
    inspikes = []
    outspikes = []
    c = 0
    for i in range(trials):

        trace = []
        gtrace = []
        for inp in inp1:
            trace.append(inp.generate(deltaT))
            gtrace += [x.G for x in inp.outputs]
        for inp in inp2:
            trace.append(inp.generate(deltaT))
            gtrace += [x.G for x in inp.outputs]

 #       intrace.append(gtrace)
 #       inspikes.append(trace)
        for neuron in hidden_layer:
            neuron.Update(deltaT)
        O = outputneuron.Update(deltaT)

        if(O):
            c+=1
            print "wow an outputspike! ",c

#        outspikes.append(O)
#        outtrace.append(outputneuron.V)


        h = (2.0*correct-1.0)*(2*O-1)
        h = (2.0*correct - 1.0)*O
        if(h!= 0):
            print "h: ",h
#        h = O
        for synapse in synapses:
            synapse.Update(h,0.3)


    return intrace,outtrace,inspikes,outspikes




def recordsignal(x,y,inp1,inp2,hidden_layer,outputneuron,synapses,deltaT,trials):

    correct = (x*y)%2
    print "correct: ",correct
    for inp in inp1:
        inp.setstate(x)
    for inp in inp2:
        inp.setstate(y)


        
    intrace = []
    outtrace = []
    inspikes = []
    outspikes = []
    c = 0
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

        if(O):
            c+=1
            print "wow an outputspike! ",c

        outspikes.append(O)
        outtrace.append(outputneuron.V)


        h = (2.0*correct-1.0)*(2*O-1)
        h = (2.0*correct - 1.0)*O
#        if(h!= 0):
#            print "h: ",h
#        h = O
#        for synapse in synapses:
#            synapse.Update(h,0.3)


    return intrace,outtrace,inspikes,outspikes



def trainsignal(x,y,inp1,inp2,hidden_layer,outputneuron,synapses,deltaT,trials):

    correct = (x*y)%2

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

        if(O):
            print "omg a spike!"
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
                for synapse in synapses:
                    synapse.G = 0
                    synapse.trace_e = 0
                int,outt,ins,outs = testsignal(x,y,inp1,inp2,hidden_layer,outputneuron,synapses,deltaT,trials)
                intrace += int
                outtrace += outt
                inspikes += ins
                outspikes += outs
                print "\ttrained "+str(x)+str(y)
#        fp = open("progressfile","w")
#        dump([intrace,outtrace,inspikes,outspikes,[synapse.q for synapse in outputneuron.inputs]],fp)
#        fp.close()
        print "finished epoch ",e

def test(inp1,inp2,hidden_layer,outputneuron,synapses,deltaT,trials):
    intrace = []
    outtrace = []
    inspikes = []
    outspikes = []
    for x in [0,1]:
        for y in [0,1]:
            int,outt,ins,outs = recordsignal(x,y,inp1,inp2,hidden_layer,outputneuron,synapses,deltaT,trials)
            for synapse in synapses:
                synapse.trace_e = 0.0
                
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
    train(*tuple(list(vars)+[1.0/2000.0,1000, 200]))

    intrace,outtrace,inspikes,outspikes,qvals = test(*tuple(list(vars)+[ 1.0/2000.0, 100]))

    fp = open("neurondump_and","w")
    
    dump([intrace,outtrace,inspikes,outspikes,qvals],fp)
    fp.close()


if __name__ == "__main__":
    main()
