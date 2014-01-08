
import NEF
from matplotlib import pyplot as plt


synapse = NEF.Synapse(initialQ = 0.0)

neuron = NEF.NEFneuron(synapse = synapse)

layer = NEF.NEF_layer(layer = [neuron],tau_PSC = 0.000001)

deltaT = 0.5*NEF.ms

xvals = [-1.0 + 0.1*x for x in range(20)]

yvals = [neuron.a(x) for x in xvals]

x = 1.0

total = 0
print 3/deltaT
for a in range(int(3/deltaT)):
    total += layer.Process(x,deltaT)

print "total: ",total/3


