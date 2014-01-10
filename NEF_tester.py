
import NEF
from matplotlib import pyplot as plt
from random import choice
from math import log,exp
from random import normalvariate,random

def Error(p,target,deltaT):
    Error.value = Error.value*exp(-deltaT/Error.tau)
    Error.value += Error.grace - (target - p)**2


    return Error.value

Error.grace = 0.00
Error.value = 0.0
Error.tau = 0.1*NEF.ms




#synapses = [NEF.Synapse(inhibitory = (x%2)*2-1,initialQ = 0.0) for x in range(1000)]

synapses = [NEF.Synapse(inhibitory = choice([-1,1]),initialQ = 0.0) for x in range(500)]

#neurons = [NEF.NEFneuron(synapse = x) for x in synapses]
neurons = [NEF.NEFneuron(synapse = x,e = choice([-1,1]),alpha = normalvariate(17*NEF.nA,3*NEF.nA),J_bias = normalvariate(10*NEF.nA,2*NEF.nA),tau_ref = normalvariate(1*NEF.ms,0.3*NEF.ms),tau_RC = normalvariate(20*NEF.ms,4*NEF.ms)) for x in synapses]

layer = NEF.NEF_layer(layer = neurons,tau_PSC = 10* NEF.ms)

deltaT = 0.5*NEF.ms

feedbackrate = 10000
eta = 0.2
x = 1.0

total = 0
print 3/deltaT
tvals = []
xhatvals = []

x = 0.4
for a in range(int(3/deltaT)):
    tvals.append(a*deltaT)
    xhatvals.append(layer.Process(x,deltaT))

plt.plot(tvals,xhatvals)


plt.show()
c = 0
while(1):
    c+=1
    for a in range(100):
        x = choice([-1,1])*0.4
        print "c: ",c
        print "iteration: ",a
        print "trying x=",x
        for z in range(int(1.5/deltaT)):
            val = layer.Process(x,deltaT)
            er = Error(val,x,deltaT)
            if(random() < deltaT*feedbackrate):

                layer.Update(er,eta)


    x = 0.4
    tvals = []
    xhatvals = []
    ervals = []
    print "average q: ",reduce(lambda x,y:x+y,[neuron.synapse.q for neuron in layer.layer],0)/len(layer.layer)
    for a in range(int(0.5/deltaT)):
        tvals.append(a*deltaT)
        val = layer.Process(x,deltaT)
        xhatvals.append(val)
        ervals.append(eta*Error(val,x,deltaT))
    plt.title("x: "+str(x))
    plt.plot(tvals,xhatvals)
    plt.plot(tvals,ervals)
    plt.savefig("datafig_"+str(c))
    c+=1
    for a in range(100):
        x = choice([-1,1])*0.4
        print "c: ",c
        print "iteration: ",a
        print "trying x=",x
        for z in range(int(1.5/deltaT)):
            val = layer.Process(x,deltaT)
            er = Error(val,x,deltaT)
            if(random() < deltaT*feedbackrate):
                layer.Update(er,eta)


    x = -0.4
    tvals = []
    xhatvals = []
    ervals = []
    print "average q: ",reduce(lambda x,y:x+y,[neuron.synapse.q for neuron in layer.layer],0)/len(layer.layer)
    for a in range(int(0.5/deltaT)):
        tvals.append(a*deltaT)
        val = layer.Process(x,deltaT)
        xhatvals.append(val)
        ervals.append(eta*Error(val,x,deltaT))
    plt.title("x: "+str(x))
    plt.plot(tvals,xhatvals)
    plt.plot(tvals,ervals)
    plt.savefig("datafig_"+str(c))

