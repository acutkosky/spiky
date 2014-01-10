
import NEF
from matplotlib import pyplot as plt
from random import choice
from math import log,exp
from random import normalvariate,random
from pickle import dump,load

def Error(p,target,deltaT):
    Error.value = Error.value*exp(-deltaT/Error.tau)
    Error.value += Error.grace - (target - p)**2


    return Error.value

def target(x):
    return x

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
Error.tau = 0.1*NEF.ms




#synapses = [NEF.Synapse(inhibitory = (x%2)*2-1,initialQ = 0.0) for x in range(1000)]

synapses = [NEF.Synapse(inhibitory = choice([-1,1]),initialQ = 0.0) for x in range(500)]

#neurons = [NEF.NEFneuron(synapse = x) for x in synapses]
<<<<<<< HEAD
neurons = [NEF.NEFneuron(synapse = x,e = choice([-1,1]),alpha = normalvariate(17*NEF.nA,3*NEF.nA),J_bias = normalvariate(10*NEF.nA,2*NEF.nA),tau_ref = normalvariate(1*NEF.ms,0.3*NEF.ms),tau_RC = normalvariate(20*NEF.ms,4*NEF.ms)) for x in synapses]
=======
neurons = [NEF.NEFneuron(synapse = x,e = choice([-1,1]),alpha = normalvariate(16*NEF.nA,5*NEF.nA),J_bias = normalvariate(10*NEF.nA,10*NEF.nA),tau_ref = normalvariate(1.2*NEF.ms,0.3*NEF.ms),tau_RC = normalvariate(20*NEF.ms,4*NEF.ms),J_th = normalvariate(1*NEF.nA,.2*NEF.nA)) for x in synapses]
>>>>>>> 625db5af8125a9b1988b257a93f3da7c546f3083

layer = NEF.NEF_layer(layer = neurons,tau_PSC = 10* NEF.ms)

#fp = open("neflayer_allpoints")
#layer = load(fp)
#fp.close()

deltaT = 0.5*NEF.ms

feedbackrate = 10000
eta = 0.2
x = 1.0

total = 0
print 3/deltaT
tvals = []
xhatvals = []


xvals = [x*0.1 for x in range(-20,20)]
for i in range(100):
    plottuning(choice(neurons),xvals)

plt.show()
<<<<<<< HEAD
c = 0
while(1):
    c+=1
    for a in range(100):
        x = choice([-1,1])*0.4
        print "c: ",c
        print "iteration: ",a
        print "trying x=",x
        for z in range(int(1.5/deltaT)):
=======

#initplot(layer)

#plt.savefig("dataplot0_slowrate")
#plt.show()
c = 0
while(1):
    c+=1
    for a in range(1):
        x = choice([-2,-1,1,2])*0.2#random()*2.0-1.0
        x = random()*4.0-2.0
        x = choice([-2,-1,0,1,2])
        x = 0.4
        t = target(x)
        print "epoch: ",c
        print "iteration: ",a
        print "trying x= "+str(x)+" target is: "+str(t)
        for z in range(int(5.0/deltaT)):
>>>>>>> 625db5af8125a9b1988b257a93f3da7c546f3083
            val = layer.Process(x,deltaT)
            er = Error(val,t,deltaT)
            if(random() < deltaT*feedbackrate):
                layer.Update(er,eta)

    fp = open("neflayer_5points_id_doublerange_morevariation","w")
    dump(layer,fp)
    fp.close()
    x = choice([-2,-1,1,2])*0.2#random()*2.0-1.0
    x = random()*4.0-2.0
    x = choice([-2,-1,0,1,2])
    x = 0.4
    t = target(x)
    tvals = []
    xhatvals = []
    ervals = []
    print "average q: ",reduce(lambda x,y:x+y,[neuron.synapse.q for neuron in layer.layer],0)/len(layer.layer)
    for a in range(int(0.5/deltaT)):
        tvals.append(a*deltaT)
        val = layer.Process(x,deltaT)
        xhatvals.append(val)
<<<<<<< HEAD
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

=======
        ervals.append(eta*Error(val,t,deltaT))
    plt.clf()
    plt.title("xvalue = "+str(x)+" target = "+str(t))
    plt.plot(tvals,xhatvals)
    plt.plot(tvals,ervals)
    plt.show()
    plt.savefig("dataplot_"+"5points_id_doublerange_morevatiation"+str(c))    


>>>>>>> 625db5af8125a9b1988b257a93f3da7c546f3083
