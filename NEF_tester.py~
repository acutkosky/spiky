
import NEF
from matplotlib import pyplot as plt
from random import choice

synapses = [NEF.Synapse(inhibitory = (x%2)*2-1,initialQ = 0.0) for x in range(500)]

synapses = [NEF.Synapse(inhibitory = choice([1,1,1,1]),initialQ = 9.0) for x in range(200)]

neurons = [NEF.NEFneuron(synapse = x) for x in synapses]

layer = NEF.NEF_layer(layer = neurons,tau_PSC = 10* NEF.ms)

deltaT = 0.5*NEF.ms


x = 1.0

total = 0
print 3/deltaT
tvals = []
xhatvals = []
for a in range(int(3/deltaT)):
    tvals.append(a*deltaT)
    xhatvals.append(layer.Process(x,deltaT))

plt.plot(tvals,xhatvals)


plt.show()
