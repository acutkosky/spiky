import NEF
from sparseNEF import SparseNEF
from random import random,normalvariate,expovariate
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt,cos,sin

def SQError(p,target,deltaT):
    return -(target-p)**2

def weight_histogram(layer,binnum=None):
    weights = [reduce(lambda x,synapse:x+synapse.inhibitory*synapse.Pval(),neuron.synapses,0) for neuron in layer.layer]

    if(binnum == None):
        plt.hist(weights,normed = True)
    else:
        plt.hist(weights,bins=binnum,normed = True)




def valtopair(x):
    if(x>0):
        c = x+random()*(2000-2*x)
    else:
        c = -x + random()*(2000+2*x)
#    c = 400+400*random()

    fp = (c+x)/2.0
    fm = (c-x)/2.0
    try:
        assert(fp>=0 and fm>=0)
    except:
        print "c: ",c," x: ",x
        exit()
    return (fp,fm)

def pairtoval(pair):
    return pair[0]-pair[1]



def plotrange(f,xmin,xmax,resolution,alabel = None):
    xvals = [x/float(resolution) for x in range(resolution*xmin,resolution*xmax)]
    plt.plot(xvals,map(f,xvals),label = alabel)

def plotavs(layer,xmin,xmax,resolution,savename = None,display = True,title = ""):
    plt.clf()

#    plotrange(lambda x:layer.getaverage(valtopair(x)),xmin,xmax,"decoded values")
    xvals = []
    cms = []
    decvals = []
    dec_pairp=[]
    dec_pairm=[]
    dec_cms = []
    for x in range(resolution*xmin,resolution*xmax):
        pair = valtopair(x/float(resolution))
        xvals.append(x/float(resolution))
        dec = layer.getaverage(pair)
        decvals.append(dec)
        dec_pair = layer.getpair(pair)
        dec_pairp.append(-dec_pair[0])
        dec_pairm.append(-dec_pair[1])

        cms.append((pair[0]+pair[1])/2.0)
    plt.plot(xvals,decvals,label = "decoded values")
    plt.plot(xvals,cms,label="input common mode")
    plt.plot(xvals,dec_pairp,label="output f+")
    plt.plot(xvals,dec_pairm,label="output f-")
                    


#    plotrange(layer.getCM,xmin,xmax,"common modes")
    plotrange(lambda x: target(valtopair(x)),xmin,xmax,resolution,"target values")
#    plotrange(lambda x:Error(layer.getaverage(valtopair(x)),target(valtopair(x)),1),xmin,xmax,resolution,"error")
    ervals = [SQError(layer.getaverage(valtopair(x)),target(valtopair(x)),1) for x in [x/float(resolution) for x in range(resolution*xmin,resolution*xmax)]]
 
    avsq = reduce(lambda x,y:x+y**2,ervals)/len(ervals)
    
    avsq = 0
    for er in ervals:
        avsq += er
    
    avsq = avsq/len(ervals)

    rms = sqrt(-avsq)

    if(title != ""):
        title += " RMS Error: "+str(rms)
    else:
        title = "RMS Error: "+str(rms)

    plt.title(title)

    plt.legend(loc=2)
    if(savename != None):
        plt.savefig(savename)
    if(display):
        plt.show()




def sparseRMSE(sparsenet,target,xvals):
    tvals = np.array([target(x) for x in xvals])
    pvals = np.array([sparsenet.GetVal(x) for x in xvals])

    return sqrt(reduce(lambda x,y: x+y,(tvals-pvals)**2,0)/len(xvals))

def normalRMSE(neflayer,target,xvals):
    tvals = np.array([target(x) for x in xvals])
    pvals = np.array([neflayer.GetAverage(x) for x in xvals])

    return sqrt(reduce(lambda x,y: x+y,(tvals-pvals)**2,0)/len(xvals))
    
    
def plotsparseavs(sparsenet,target,xmin,xmax,resolution,savename = None,display = True,title = ""):
    plt.clf()

    xvals = []
    cms = []
    decvals = []
    dec_pairp=[]
    dec_pairm=[]
    dec_cms = []
    for x in range(resolution*xmin,resolution*xmax):
        xval = x/float(resolution)
        xvals.append(x/float(resolution))
        dec = sparsenet.GetVal(xval)
        decvals.append(dec)
        dec_pair = sparsenet.getpair(xval)
        dec_pairp.append(-dec_pair[0])
        dec_pairm.append(-dec_pair[1])


    plt.plot(xvals,decvals,label = "decoded values")

#    plt.plot(xvals,dec_pairp,label="output f+")
#    plt.plot(xvals,dec_pairm,label="output f-")
                    



    plotrange(lambda x: target(x),xmin,xmax,resolution,"target values")

    ervals = [SQError(sparsenet.GetVal(x),target(x),1) for x in [x/float(resolution) for x in range(resolution*xmin,resolution*xmax)]]
 
    avsq = reduce(lambda x,y:x+y**2,ervals)/len(ervals)
    
    avsq = 0
    for er in ervals:
        avsq += er
    
    avsq = avsq/len(ervals)

    rms = sqrt(-avsq)

    if(title != ""):
        title += " RMS Error: "+str(rms)
    else:
        title = "RMS Error: "+str(rms)

    plt.title(title)

    plt.legend(loc=2)
    if(savename != None):
        plt.savefig(savename)
    if(display):
        plt.show()



def plottuning(neuron,xvals):
    yvals = [neuron.a(x) for x in xvals]
    vallist = zip(map(pairtoval,xvals),yvals)
    vallist.sort(key = lambda x:x[0])
    xvals = [x[0] for x  in vallist]
    yvals = [x[1] for x in vallist]
    plt.plot(xvals,yvals)

def initplot(layer):
    x = -2
    t = target(x)
    for a in range(int(0.5/deltaT)):
        tvals.append(a*deltaT)
        xhatvals.append(layer.Process(x,deltaT))
    x = -1
    for a in range(int(0.5/deltaT)):
        tvals.append(0.5+a*deltaT)
        xhatvals.append(layer.Process(x,deltaT))
    x = 1
    for a in range(int(0.5/deltaT)):
        tvals.append(1.0+a*deltaT)
        xhatvals.append(layer.Process(x,deltaT))
    x = 2
    for a in range(int(0.5/deltaT)):
        tvals.append(1.5+a*deltaT)
        xhatvals.append(layer.Process(x,deltaT))

    plt.plot(tvals,xhatvals)
    plt.show()




def randunit(d):
#    v = 2
    v = np.array([normalvariate(0,10) for x in range(d)])
#    while(np.linalg.norm(v)>1):
#        print "trying!"

#random()*2-1.0 for x in range(d)])

    
    return v/np.linalg.norm(v)

def randweighting(d):
    var = 0.3
#    var = 0.0
    return np.array([var*(2*random()-1.0)+1.0,-(var*(2*random()-1.0)+1.0)])






def createlayer(num = 100,dim = 1):

    layersize = num#100
    weight_val = 1#(10*NEF.ms)
    inhibsynapses = [NEF.Synapse(inhibitory = -1,initialQ = 0*(random()-0.5)-4.0) for x in range(layersize)]
    excitsynapses = [NEF.Synapse(inhibitory = 1,initialQ = 0*(random()-0.5)-4.0) for x in range(layersize)]
    
    
#    neurons = [NEF.NEFneuron(synapses = [excitsynapses[i],inhibsynapses[i]],e = randunit(dim)*randweighting(2),alpha = (1.0/400.0)*normalvariate(17*NEF.nA,5*NEF.nA),J_bias = normalvariate(10*NEF.nA,5*NEF.nA),tau_ref = normalvariate(1.5*NEF.ms,0.3*NEF.ms),tau_RC = normalvariate(20*NEF.ms,4*NEF.ms),J_th = normalvariate(1*NEF.nA,.2*NEF.nA)) for i in range(layersize)]

    neurons = [NEF.NEFneuron(synapses = [excitsynapses[i],inhibsynapses[i]],e = randunit(dim),alpha = (1.0/400.0)*normalvariate(17*NEF.nA,5*NEF.nA),J_bias = (random()*2-1.0)*20*NEF.nA+7*NEF.nA,tau_ref = normalvariate(1.5*NEF.ms,0.3*NEF.ms),tau_RC = normalvariate(20*NEF.ms,4*NEF.ms),J_th = normalvariate(1*NEF.nA,.2*NEF.nA)) for i in range(layersize)]
    
    layer = NEF.NEF_layer(layer = neurons,tau_PSC = 10 * NEF.ms,weight = weight_val)
    return layer


def createsparselayer(num1 = 100,num2 = 100,dim = 1,numconnections = None ):

    if(numconnections == None):
        numconnections = 30*dim
    weight_val = 1

    inhibsynapses = [NEF.Synapse(inhibitory = -1,initialQ = 0*(random()-0.5)-4.0) for x in range(num1)]
    excitsynapses = [NEF.Synapse(inhibitory = 1,initialQ = 0*(random()-0.5)-4.0) for x in range(num1)]

    layer1 = [NEF.NEFneuron(synapses = [excitsynapses[i],inhibsynapses[i]],e = randunit(dim),alpha = (1.0/400.0)*normalvariate(17*NEF.nA,5*NEF.nA),J_bias = (random()*2-1.0)*20*NEF.nA+7*NEF.nA,tau_ref = normalvariate(1.5*NEF.ms,0.3*NEF.ms),tau_RC = normalvariate(20*NEF.ms,4*NEF.ms),J_th = normalvariate(1*NEF.nA,.2*NEF.nA)) for i in range(num1)]


    inhibsynapses = [NEF.Synapse(inhibitory = -1,initialQ = 0*(random()-0.5)-4.0) for x in range(num2)]
    excitsynapses = [NEF.Synapse(inhibitory = 1,initialQ = 0*(random()-0.5)-4.0) for x in range(num2)]

    dim = 1
    layer2 = [NEF.NEFneuron(synapses = [excitsynapses[i],inhibsynapses[i]],e = randunit(dim),alpha = (1.0/400.0)*normalvariate(17*NEF.nA,5*NEF.nA),J_bias = (random()*2-1.0)*20*NEF.nA+7*NEF.nA,tau_ref = normalvariate(1.5*NEF.ms,0.3*NEF.ms),tau_RC = normalvariate(20*NEF.ms,4*NEF.ms),J_th = normalvariate(1*NEF.nA,.2*NEF.nA)) for i in range(num2)]


    return SparseNEF(layer1,layer2,numconnections)


