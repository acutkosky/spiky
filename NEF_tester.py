
import NEF
from matplotlib import pyplot as plt
from random import choice
from math import log,exp,sqrt,sin
from random import normalvariate,random
from pickle import dump,load
#from math import abs
import numpy as np

def Error(p,target,deltaT):
    Error.value = Error.value*exp(-deltaT/Error.tau)
#    Error.value += Error.grace - abs(target - p)
    Error.value += Error.grace - (target - p)**2
    return Error.value

def sigmoid(er):
    return er
#    return (2.0/(1.0+exp(-2*er))-1.0)
targetname = "target"



def valtopair(x):
    c = 400+400*random()

    fp = (c+x)/2.0
    fm = (c-x)/2.0
    return (fp,fm)

def pairtoval(pair):
    return pair[0]-pair[1]

def target(x):
    global targetname 
    targetname = "sin"
    z = x[0]-x[1]
#    return -z*z
    return 400.0*sin(3.14159264/400.0*z)

def plotrange(f,xmin,xmax,resolution,alabel = None):
    xvals = [x/float(resolution) for x in range(resolution*xmin,resolution*xmax)]
    plt.plot(xvals,map(f,xvals),label = alabel)

def plotavs(layer,xmin,xmax,resolution,savename = None,display = True,title = ""):
    plt.clf()

#    plotrange(lambda x:layer.getaverage(valtopair(x)),xmin,xmax,"decoded values")
    xvals = []
    cms = []
    decvals = []
    for x in range(resolution*xmin,resolution*xmax):
        pair = valtopair(x/float(resolution))
        xvals.append(x/float(resolution))
        decvals.append(layer.getaverage(pair))
        cms.append((pair[0]+pair[1])/2.0)
    plt.plot(xvals,decvals,label = "decoded values")
    plt.plot(xvals,cms,label="common mode")
           
                    


#    plotrange(layer.getCM,xmin,xmax,"common modes")
    plotrange(lambda x: target(valtopair(x)),xmin,xmax,resolution,"target values")
#    plotrange(lambda x:Error(layer.getaverage(valtopair(x)),target(valtopair(x)),1),xmin,xmax,resolution,"error")
    ervals = [Error(layer.getaverage(valtopair(x)),target(valtopair(x)),1) for x in [x/float(resolution) for x in range(resolution*xmin,resolution*xmax)]]
 
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


Error.grace = 0.00
Error.value = 0.0
Error.tau = 0.01*NEF.ms


def randunit(d):
    v = 2
    while(np.linalg.norm(v)>1):
        v = np.array([random()*2-1.0 for x in range(d)])
    
    return v/np.linalg.norm(v)

def randweighting(d):
    var = 10.0
    var = 0.0
    return np.array([var*(2*random()-1.0)+1.0,-(var*(2*random()-1.0)+1.0)])

#synapses = [NEF.Synapse(inhibitory = (x%2)*2-1,initialQ = 0.0) for x in range(1000)]


layersize = 100
weight_val = 1#(10*NEF.ms)**2
inhibsynapses = [NEF.Synapse(inhibitory = -1,initialQ = random()-0.5-4.0) for x in range(layersize)]
excitsynapses = [NEF.Synapse(inhibitory = 1,initialQ = random()-0.5-4.0) for x in range(layersize)]

#neurons = [NEF.NEFneuron(synapse = x) for x in synapses]
neurons = [NEF.NEFneuron(synapses = [excitsynapses[i],inhibsynapses[i]],e = choice([-1,1])*randweighting(2),alpha = (1.0/400.0)*normalvariate(16*NEF.nA,5*NEF.nA),J_bias = normalvariate(10*NEF.nA,15*NEF.nA),tau_ref = normalvariate(1.5*NEF.ms,0.3*NEF.ms),tau_RC = normalvariate(20*NEF.ms,4*NEF.ms),J_th = normalvariate(1*NEF.nA,.2*NEF.nA)) for i in range(layersize)]

layer = NEF.NEF_layer(layer = neurons,tau_PSC = 10* NEF.ms,weight = weight_val)

#fp = open("neflayer_allpoints")
#layer = load(fp)
#fp.close()

deltaT = 0.5*NEF.ms

feedbackrate =0.5
updaterate = 5
eta = 0.001
targetx = 1.0
x = 0.4
time = 0.25

total = 0
print 3/deltaT
tvals = []
xhatvals = []


#xvals = [x*0.01 for x in range(-200,200)]
res = 100.0
xvals = [(x*1.0/res,y*1.0/res) for x in range(0,int(res)) for y in range(0,int(res))]
xvals = [valtopair(400.0*(2*random()-1.0)) for x in range(1000)]
for i in range(100):
    plottuning(choice(neurons),xvals)

plt.show()

#NEF.LeastSquaresSolve(xvals,target,layer)



plotavs(layer,-400,400,1)
#plotavs(layer,-400,400,1)
#plotavs(layer,-400,400,1)
#exit()
#initplot(layer)

#plt.savefig("dataplot0_slowrate")
#plt.show()
c = 0
pltcount = 0
while(1):
    c+=1
    for a in range(1):
        x = choice([-2,-1,1,2])*0.2#random()*2.0-1.0
        x = random()*4.0-2.0
        x = choice([-2,-1,0,1,2])
        x = targetx
        x = 1.0
        if(c%2):
            x = 0.4
        x = 400.0*(random()*2.0-1.0)

        pair = valtopair(x)
        t = target(pair)
        display = (c%int(30/time) == 0)
        print "epoch: ",c
        print "iteration: ",a
        print "trying x= "+str(x)+" target is: "+str(t)
        etot = 0
        xtot = 0
        avxtot = 0
        count = 0
        tvals = []
        xhatvals = []
        ervals = []
        avvals = []
        aver = 0.0
        averc = 0
 #       display = True
        print "display: ",display
        for z in range(int(time/deltaT)):
            val = layer.Process(pair,deltaT)
            xtot += val
            avxtot += layer.average
            er = sigmoid(Error(val,t,deltaT))
            layer.RecordErr(er)
            aver += er
            averc += 1
            if(display):
                tvals.append(a*1.0+z*deltaT)
                xhatvals.append(val)
                avvals.append(layer.average)
                ervals.append(er*eta)
            etot += er
            count += 1
            if(c%int(updaterate/time)==0 and z ==0):#random() < deltaT*feedbackrate):
                print "updating!"
                layer.RecUpdate(eta)
#                layer.Update(aver/averc,eta)
                aver = 0
                averc = 0
        print "average error: ",etot/count
        print "average x: ",xtot/count
        print "predicted average: ",avxtot/count
        print "average q: ",reduce(lambda x,y:x+y,[reduce(lambda
x,y:x+y.q,neuron.synapses,0) for neuron in layer.layer],0)/len(layer.layer)

        if(display):
            pltcount += 1
            plt.clf()
            plt.title("xvalue = "+str(x)+" target = "+str(t))
            v = "1p0"
            if (x==0.4):
                v = "0p4"
            plt.plot(tvals,xhatvals,label="decoded")
#            plt.plot(tvals,ervals,label ="error")
            plt.plot(tvals,avvals,label="a vals")
            plt.legend()
#            plt.show()
#            plt.savefig("savedfig_allpoints_normalized_300neurons_etap05_woverallplots_"+str(c))
                #        plt.savefig("savedfig_both_"+v+"_wsigmoid_m3_"+str(c))

            savename = ("figs/savedgraph_frequencies_allpoints_normalized_"+str(time)+"perval_"+str(updaterate)+"updaterate_"+targetname+"_"+str(layersize)+"neurons_update"+str(feedbackrate)+"_eta"+str(eta)+"_weight"+str(weight_val)+"_aver_clearerr_"+str(pltcount)).replace(".","p")
            print "saving to: "+savename+".png"
            #plt.show()
            plotavs(layer,-400,400,1,savename,display = False)
#            plt.show()
            
#    fp = open("neflayer_5points_id_doublerange_morevariation","w")
#    dump(layer,fp)
#    fp.close()
#    x = choice([-2,-1,1,2])*0.2#random()*2.0-1.0
#    x = random()*4.0-2.0
#    x = choice([-2,-1,0,1,2])
#    x = targetx
#    t = target(x)
#    tvals = []
#    xhatvals = []
#    ervals = []

#    for a in range(int(0.5/deltaT)):
#        tvals.append(a*deltaT)
#        val = layer.Process(x,deltaT)
#        xhatvals.append(val)
#        ervals.append(eta*Error(val,t,deltaT))
#    plt.clf()
#    plt.title("xvalue = "+str(x)+" target = "+str(t))
#    plt.plot(tvals,xhatvals)
#    plt.plot(tvals,ervals)
#    plt.show()
#    plt.savefig("dataplot_"+"5points_id_doublerange_morevatiation"+str(c))    


