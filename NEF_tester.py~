
import NEF
from matplotlib import pyplot as plt
from random import choice
from math import log,exp
from random import normalvariate,random

def Error(p,target,deltaT):
    Error.value += Error.grace - (target - p)**2
    Error.value = Error.value*exp(deltaT/Error.tau)
    return Error.value

Error.grace = 0.01
Error.value = 0.0
Error.tau = 10*NEF.ms




#synapses = [NEF.Synapse(inhibitory = (x%2)*2-1,initialQ = 0.0) for x in range(1000)]

synapses = [NEF.Synapse(inhibitory = choice([-1,1]),initialQ = 0.0) for x in range(1500)]

#neurons = [NEF.NEFneuron(synapse = x) for x in synapses]
neurons = [NEF.NEFneuron(synapse = x,e = choice([-1,1]),alpha = normalvariate(17*NEF.nA,.3*NEF.nA),J_bias = normalvariate(10*NEF.nA,.2*NEF.nA),tau_ref = normalvariate(1*NEF.ms,0.03*NEF.ms),tau_RC = normalvariate(20*NEF.ms,.4*NEF.ms)) for x in synapses]

layer = NEF.NEF_layer(layer = neurons,tau_PSC = 10* NEF.ms)

deltaT = 0.5*NEF.ms

feedbackrate = 100

x = 1.0

total = 0
print 3/deltaT
tvals = []
xhatvals = []

x = 0.5
for a in range(int(3/deltaT)):
    tvals.append(a*deltaT)
    xhatvals.append(layer.Process(x,deltaT))

plt.plot(tvals,xhatvals)


plt.show()


for a in range(10):
    x = 0.5
    print "iteration: ",a
    print "trying x=",x
    for z in range(int(0.5/deltaT)):
        val = layer.Process(x,deltaT)
        er = Error(val,x,deltaT)
        if(random() < deltaT*feedbackrate):
            print "updating!",er
            layer.Update(er,0.2)


x = 0.5
tvals = []
xhatvals = []
for a in range(int(3/deltaT)):
    tvals.append(a*deltaT)
    xhatvals.append(layer.Process(x,deltaT))

plt.plot(tvals,xhatvals)


plt.show()
