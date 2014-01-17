
import NEF
from matplotlib import pyplot as plt
from random import choice
from math import log,exp,sqrt
from random import normalvariate,random
from pickle import dump,load
#from math import abs

def Error(p,target,deltaT):
    Error.value = Error.value*exp(-deltaT/Error.tau)
#    Error.value += Error.grace - abs(target - p)
    Error.value += Error.grace - (target - p)**2


    return Error.value



def sigmoid(er):
    return er
#    return (2.0/(1.0+exp(-2*er))-1.0)

def target(x):
    return -x*x

def plotrange(f,xmin,xmax,alabel = None):
    xvals = [x/40.0 for x in range(40*xmin,40*xmax)]
    plt.plot(xvals,map(f,xvals),label = alabel)

def plotavs(layer,xmin,xmax,savename = None,display = True,title = ""):
    plt.clf()
    plotrange(layer.getaverage,xmin,xmax,"decoded values")
    plotrange(target,xmin,xmax,"target values")
    plotrange(lambda x:Error(layer.getaverage(x),target(x),1),xmin,xmax,"error")
    ervals = [Error(layer.getaverage(x),target(x),1) for x in [x/40.0 for x in range(40*xmin,40*xmax)]]

    avsq = reduce(lambda x,y:x+y**2,ervals)/len(ervals)
    
    avsq = 0
    for er in ervals:
        avsq += er*er
    
    avsq = avsq/len(ervals)

    rms = sqrt(avsq)

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




#synapses = [NEF.Synapse(inhibitory = (x%2)*2-1,initialQ = 0.0) for x in range(1000)]

synapses = [NEF.Synapse(inhibitory = choice([-1,1]),initialQ = random()*2-1.0) for x in range(800)]

#neurons = [NEF.NEFneuron(synapse = x) for x in synapses]
neurons = [NEF.NEFneuron(synapse = x,e = choice([-1,1]),alpha = normalvariate(16*NEF.nA,5*NEF.nA),J_bias = normalvariate(10*NEF.nA,15*NEF.nA),tau_ref = normalvariate(1.5*NEF.ms,0.3*NEF.ms),tau_RC = normalvariate(20*NEF.ms,4*NEF.ms),J_th = normalvariate(1*NEF.nA,.2*NEF.nA)) for x in synapses]

layer = NEF.NEF_layer(layer = neurons,tau_PSC = 10* NEF.ms)

#fp = open("neflayer_allpoints")
#layer = load(fp)
#fp.close()

deltaT = 0.5*NEF.ms

feedbackrate =1000
eta = 0.1
targetx = 1.0
x = 0.4

total = 0
print 3/deltaT
tvals = []
xhatvals = []


xvals = [x*0.1 for x in range(-20,20)]
for i in range(100):
    plottuning(choice(neurons),xvals)

plt.show()

plotavs(layer,-1,1)
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
        x = random()*2.0-1.0
        t = target(x)
        display = (c%30 == 0)
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
        print "display: ",display
        for z in range(int(2.0/deltaT)):

            val = layer.Process(x,deltaT)
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
            if(random() < deltaT*feedbackrate):
                layer.RecUpdate(eta)
#                layer.Update(aver/averc,eta)
                aver = 0
                averc = 0
        print "average error: ",etot/count
        print "average x: ",xtot/count
        print "predicted average: ",avxtot/count
        print "average q: ",reduce(lambda x,y:x+y,[neuron.synapse.q for neuron in layer.layer],0)/len(layer.layer)

        if(display):
            pltcount += 1
            plt.clf()
            plt.title("xvalue = "+str(x)+" target = "+str(t))
            v = "1p0"
            if (x==0.4):
                v = "0p4"
            plt.plot(tvals,xhatvals)
            plt.plot(tvals,ervals)
            plt.plot(tvals,avvals)

#            plt.savefig("savedfig_allpoints_normalized_300neurons_etap05_woverallplots_"+str(c))
                #        plt.savefig("savedfig_both_"+v+"_wsigmoid_m3_"+str(c))
            savename = "figs/savedgraph_allpoints_normalized_800neurons_invquadratic_update1000_eta0p1_aver_clearerr_"+str(pltcount)
            print "saving to: "+savename+".png"
            plotavs(layer,-1,1,savename,display = False)
#            plt.show()
            
#    fp = open("neflayer_5points_id_doublerange_morevariation","w")
#    dump(layer,fp)
#    fp.close()
    x = choice([-2,-1,1,2])*0.2#random()*2.0-1.0
    x = random()*4.0-2.0
    x = choice([-2,-1,0,1,2])
    x = targetx
    t = target(x)
    tvals = []
    xhatvals = []
    ervals = []

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


