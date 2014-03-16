from matplotlib import pyplot
from random import random
from scipy.stats import poisson
from math import log,sqrt
from numpy import exp
from numpy import sign
import numpy as np
from copy import deepcopy
from random import choice
#let's just block this out for now...

#ok we need the firing rate function


def getrate(x):#,tau_ref,tau_RC,J_th,alpha,J_bias):
    nA = 0.000000001
    ms = 0.001
    tau_ref = 2.5*ms
    alpha = 17*nA#/100
    J_bias = 10*nA
    tau_RC = 20*ms
    J_th = 1*nA
    if (alpha*x+J_bias <= J_th):
        return 0.0

    return 1.0/(tau_ref-tau_RC*log(1.0-J_th/(alpha*x+J_bias)))



#ok, so we generate spike trains...

def prvs(x):
    if (x <= 0.0):
        return 0.0
    return poisson.rvs(x)

pvecrvs = np.vectorize(prvs)
def sample(rates):
    s = sign(rates)
    spikes = pvecrvs(abs(rates))

    return np.multiply(s,spikes)


def GenSpikes(inputs,time):
    vgetrate = np.vectorize(getrate)
    return sample(vgetrate(inputs)*time)
#[getrate(x)/time for x in inputs])

def Synapse(spikes,weights):
    r = weights*spikes
    return sample(r)




#ok, so CD works like this:
#first assume input to visible neurons is from data
#generate visible neuron firings (genspikes)
#generate hidden neuron firings (synapse,genspikes)
#compute correlations (outerproduct)
#propogate back to visible layer (synapse,genspikes)
#send to hidden layer again (synapse,genspikes)
#compute correlations (outerproduct)
#take difference = update matrix

#repeat!


#note that this can all be vectorized - so do that!

def CD(data,weights,alpha,reg=0):
    time = 1.0/50.0
    #v0spikes = GenSpikes(data,time)
    v0spikes = sample(data*time)
    #print "v0spikes",v0spikes/time
    h0spikes = GenSpikes(Synapse(v0spikes,weights)/time,time)
    
    #print "h0spikes: ",h0spikes/time
    
    v1spikes = GenSpikes(Synapse(h0spikes,weights.transpose())/time,time)
    h1spikes = GenSpikes(Synapse(v1spikes,weights)/time,time)

    update = alpha*(v0spikes*h0spikes.transpose()-v1spikes*h1spikes.transpose()).transpose()
    #print "update: ",update
    weights += update# -weights*reg

#sample from the RBM


def SampleRBM(data,weights,samples = 1):
    time = 1.0/50.0
    av = None
    for i in range(samples):
        v0spikes = sample(data*time)#GenSpikes(data,time)
        h0spikes = GenSpikes(Synapse(v0spikes,weights)/time,time)
        v1spikes = GenSpikes(Synapse(h0spikes,weights.transpose())/time,time)
        if (av == None):
            av = deepcopy(v1spikes)
        else:
            av += v1spikes
    return av/(time*samples)



    
hsize = 20
dim = 2
weights = np.matrix([[0.3*(random()*2-1) for x in range(hsize)] for x in range(dim)]).transpose()


numIter = 1000

test = [random()*400 for x in range(100)]
test = np.matrix([deepcopy(test) for x in range(dim)])

train = [random()*400 for x in range(1000)]
train = np.matrix([deepcopy(train) for x in range(dim)])

for i in range(numIter):
    a = random()*400
    data = np.matrix([a,a]).transpose()
    #    print "data: ",data
    diff = (SampleRBM(test,weights,1)-test)
    val = (diff.transpose()*diff).trace()
    print "iteration: ",i," reconstruction error: ",sqrt(val/100.0)
    CD(train,weights,0.0001/1000,0.0)


#then pass them through the neurons, get new spike trains out


#correlate

#generate two new sets

#correlate


#update
