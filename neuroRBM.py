from matplotlib import pyplot
from random import random
from scipy.stats import poisson
from math import log,sqrt
from numpy import exp
from numpy import sign
import numpy as np
from copy import deepcopy
from random import choice
import scipy.io
import sys
from pickle import dump
#let's just block this out for now...

#ok we need the firing rate function


def printdigit(d):
    s =""
    for i in range(28):
        for j in range(28):
            s += "1" if d[j+28*i] > 0 else "0"
        s+="\n"
    print s


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
    weights += update -weights*reg

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



    
hsize = 512
dim = 784
weights = np.matrix([[0.3*(random()*2-1) for x in range(hsize)] for x in range(dim)]).transpose()

numIter = 1000

test = [random()*400 for x in range(100)]
test = np.matrix([deepcopy(test) for x in range(dim)])

train = [random()*400 for x in range(1000)]
train = np.matrix([deepcopy(train) for x in range(dim)])

train = scipy.io.loadmat('matlab_nef/trainimages.mat')['trainimages']

train = train[:,0:1]

train = train*800-400

test = scipy.io.loadmat('matlab_nef/testimages.mat')['testimages']
test = test[:,0:100]

test = test*800-400
test = train
eta = 0.001
regularization = 0.0001
savefile = sys.argv[1]
print "training with eta: ",eta," regularization: ",regularization," hidden neurons: ",hsize," savefile: ",savefile
sys.stdout.flush()

fp = open(savefile,"w")
dump(weights,fp)
fp.close()

for i in range(numIter):

    print "iteration: ",i
    if(i%10 == 0):
        diff = (SampleRBM(test,weights,10)-test)
        val = np.sum(np.multiply(diff,diff))#(diff*diff.transpose())
        print "reconstruction error: ",sqrt(val/1.0)
        rec = diff+test
        print "input: "
        printdigit(test[:,0])
        print "reconstruction: "
        printdigit(rec[:,0])
        fp = open(savefile,"w")
        dump(weights,fp)
        fp.close()
    sys.stdout.flush()
    CD(train,weights,eta/1000,regularization)


#then pass them through the neurons, get new spike trains out


#correlate

#generate two new sets

#correlate


#update
