
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
    Error.value += Error.grace - deltaT*(target - p)/Error.tau
#    if(p>target):
#        print "p: ",p
#        print "t: ",target
#        exit()
    return -(p-target)**2#exp(-(p-target)**2)#-(p/target-1)**2#-abs(p-target)#(exp(-abs(p-target)))#-(p-target)**2#+target*target#+3600

def SQError(p,target,deltaT):
    return -(target-p)**2

def sigmoid(er):
    return er
#    return (2.0/(1.0+exp(-2*er))-1.0)
targetname = "target"


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

def target(x):
    global targetname 
    targetname = "sin"
    z = x[0]-x[1]
#    return -z*z
    return 400*sin(3.141592654/400*z)#400.0*(z/400.0)#sin(3.14159264/400.0*z)

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
    v = 2
    while(np.linalg.norm(v)>1):
        v = np.array([random()*2-1.0 for x in range(d)])
    
    return v/np.linalg.norm(v)

def randweighting(d):
#    var = 0.3
    var = 0.0
    return np.array([var*(2*random()-1.0)+1.0,-(var*(2*random()-1.0)+1.0)])

#synapses = [NEF.Synapse(inhibitory = (x%2)*2-1,initialQ = 0.0) for x in range(1000)]


Error.grace = 0.0#70000.00
Error.value = 0.0
Error.tau = 0.000001*NEF.ms


layersize = 50
weight_val = 1#(10*NEF.ms)
inhibsynapses = [NEF.Synapse(inhibitory = -1,initialQ = 0*(random()-0.5)-4.0) for x in range(layersize)]
excitsynapses = [NEF.Synapse(inhibitory = 1,initialQ = 0*(random()-0.5)-4.0) for x in range(layersize)]

#neurons = [NEF.NEFneuron(synapse = x) for x in synapses]
neurons = [NEF.NEFneuron(synapses = [excitsynapses[i],inhibsynapses[i]],e = choice([-1,1])*randweighting(2),alpha = (1.0/400.0)*normalvariate(16*NEF.nA,5*NEF.nA),J_bias = normalvariate(10*NEF.nA,15*NEF.nA),tau_ref = normalvariate(1.5*NEF.ms,0.3*NEF.ms),tau_RC = normalvariate(20*NEF.ms,4*NEF.ms),J_th = normalvariate(1*NEF.nA,.2*NEF.nA)) for i in range(layersize)]

layer = NEF.NEF_layer(layer = neurons,tau_PSC = 10 * NEF.ms,weight = weight_val)

#fp = open("neflayer_allpoints")
#layer = load(fp)
#fp.close()

deltaT = 0.001#*NEF.ms

feedbackrate = 100
updaterate = 60.0#20.0#0.25
eta = 0.00004#0#0001
regularization = 0.0000001
samplefrac = 25#60
targetx = 10.0
x = 0.4
time = 2.0#
displaytime = 60
total = 0
print 3/deltaT
tvals = []
xhatvals = []
presolve = False#True

lstsq = False#True

#xvals = [x*0.01 for x in range(-200,200)]
res = 100.0
numxvals = 100
#xvals = [(x*1.0/res,y*1.0/res) for x in range(0,int(res)) for y in range(0,int(res))]
xvals = [valtopair(400.0*(2*random()-1.0)) for x in range(numxvals)]
for i in range(100):
    plottuning(choice(neurons),xvals)

plt.title("Noisy Tuning Curves")
if(lstsq):
    plt.savefig("noisytuning-"+str(numxvals)+"samples-"+str(layersize)+"neurons")
plt.show()

if(presolve):
    NEF.LeastSquaresSolve(xvals,target,layer,regularization=200)


if(lstsq):
    weight_histogram(layer,binnum=50)
    plt.savefig("weight-histogram-"+str(numxvals)+"samples-"+str(layersize)+"neurons")
    plt.show()
    plotavs(layer,-400,400,1,savename = "cm-agnostic-decode-"+str(numxvals)+"samples-"+str(layersize)+"neurons",title="Common Mode Agnostic Decode "+str(numxvals)+" pts")
    plt.show()
    exit()
else:
    plotavs(layer,-400,400,1)



#plotavs(layer,-400,400,1)
#plotavs(layer,-400,400,1)
#exit()
#initplot(layer)

#plt.savefig("dataplot0_slowrate")
#plt.show()
c = 0
pltcount = 0
erav = 0
eravcount = 0
etrack = 0
etrackcounter = 0
etracks = []
avx = 0
lastetrack = 0
esquaretrack = 0
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
#        x = 50
        pair = valtopair(x)
        t = target(pair)
#        t = 100
        display = (c%int(displaytime/time) == 0)
        if(c%500000 == 0 ):
            print "epoch: ",c
            print "iteration: ",a
            print "trying x= "+str(x)+" target is: "+str(t)+" current average is: "+str(layer.getaverage(pair))
            print "display: ",display
            
        etot = 0

        avxtot = 0
        count = 0
        tvals = []
        xhatvals = []
        ervals = []
        avvals = []
        aver = 0.0
        averc = 0
        etot_up = 0
        count_up = 0
        

#        display = True
        layer.xhat = 0
        lastx = 0

        for q in range(samplefrac):
            lastx = 0
            xtot = 0
            count = 0
            for z in range(int(time/(samplefrac*deltaT))):
                val = layer.Process(pair,deltaT)
                xtot += val/deltaT
                lastx += val/deltaT
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
                etot_up += er
                count_up += 1
            
#            if(random() <deltaT*feedbackrate):#c%int(updaterate/time)==0 and z ==0):#random() < deltaT*feedbackrate):
#              print "updating!"
#                layer.Update(-(etot/count)**2,eta)
#               layer.RecUpdate(0,0)#abs(etot_up/count_up),eta)
#               etot_up = 0
#               count_up = 0
#                layer.Update(abs(aver/averc),eta)
#              aver = 0
#             averc = 0
#            print "xtot: ",xtot/count
            erav += Error(xtot/count,t,1)

#            print "recording error: ",Error(xtot/count,t,1)
#            print "value: ",xtot/count
            avx += xtot/count
            eravcount += 1
            reperr= erav/eravcount
            etrack += layer.layer[0].synapses[0].etrackval()

            lastetrack = layer.layer[0].synapses[0].etrackval()
            esquaretrack += lastetrack**2
#           
#            print "diff: ",abs(erav/eravcount-Error(layer.getaverage(pair),t)*(-2*(1-x/float(t)))
            errorD = Error(layer.getaverage(pair),t,1)*(2*(1-layer.getaverage(pair)/float(t)))

            etracks.append(layer.layer[0].synapses[0].etrackval())
            etrackcounter += 1
            etrackval = layer.layer[0].synapses[0].etrackval()
#            print "current ratio: ",etrackval/(xtot/count - layer.getaverage(pair))
#            print (etrackval/(xtot/count - layer.getaverage(pair)) - 0.01)

#            print (etrackval/0.01 - ( xtot/count - layer.getaverage(pair)))
#            print (xtot/count - (layer.getaverage(pair)+etrackval/0.01))
#            print (Error(xtot/count,t,1) - ( Error(layer.getaverage(pair),t,1)+etrackval/0.01))
#            assert(etrackval*erav/eravcount == etrackv

#            layer.RecUpdate(Error(xtot/count,t,1),eta)
            layer.CorrectedRecUpdate(Error(xtot/count,t,1))

        if(c% int(updaterate/time)==0):
#            plt.hist(etracks)
#            plt.show()
            etracks = []
            print "updating!\n\n"
            print "time elapsed: ",c*time
            print "xval: ",x
            print "target: ",t
            print "count: ",count
            print "etrackcount: ",etrackcounter
            print "last etrack: ",lastetrack
            etrackav = etrack/etrackcounter
            etracksqav = esquaretrack/etrackcounter
            errorval = Error(layer.getaverage(pair),t,1)
            errorD = 1#-errorval*abs(layer.getaverage(pair)-t)/(layer.getaverage(pair)-t)
            print "error: ",errorval
            delta = avx/etrackcounter - layer.getaverage(pair)
            print "error plus delta: ",errorval+errorD*etrackav/0.01
            print "error (x plus delta): ",Error(layer.getaverage(pair)+etrackav/0.01,t,1)
            avgrad = etrack*errorval+esquaretrack/0.01*errorD
            print "etrack: average ",(etrack/etrackcounter)
            print "etracksquare average: ",esquaretrack/etrackcounter
            print "ratio: ",(etrackav)/(avx/etrackcounter-layer.getaverage(pair))#etrackcounter#layer.layer[0].synapses[0].etrack#etrack/etrackcounter

            print "average error: ",reperr#etot/count
            print "average x: ",avx/etrackcounter
            print "current x: ",layer.xhat/(count*deltaT)
            print "aval: ",layer.layer[0].a(pair)
            print "predicted average: ",layer.getaverage(pair)#avxtot/count
            print "pval: ",layer.layer[0].synapses[0].Pval()
            print "est grad: ",avgrad
            print "average q: ",reduce(lambda x,y:x+y,[reduce(lambda                                                             
x,y:x+y.q,neuron.synapses,0) for neuron in layer.layer],0)/len(layer.layer)

            etrack = 0
            avx = 0
            etrackcounter = 0
            esquaretrack = 0
            

            erav = 0
            eravcount = 0

            layer.CorrectedUpdate(eta,regularization)
#            layer.finalUpdate(eta)
        
        if(display):
            pltcount += 1
            savename = ("figs/savedgraph_frequencies_allpoints_correctedupdates_"+str(Error.grace)+"grace_"+str(presolve)+"presolve_"+str(displaytime)+"displaytime_"+str(time)+"perval_"+str(updaterate)+"updaterate_"+str(samplefrac)+"samplefrac_"+targetname+"_"+str(layersize)+"neurons_feedbackrate"+str(feedbackrate)+"_eta"+str(eta)+"_weight"+str(weight_val)+"_aver_clearerr_"+str(pltcount)).replace(".","p")
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


            print "saving to: "+savename+".png"
            #plt.show()
            plotavs(layer,-400,400,1,savename,display = False)
#            plt.show()

    savename = ("dumps/dump_frequencies_allpoints_correctedupdates_"+str(Error.grace)+"grace_"+str(presolve)+"presolve_"+str(displaytime)+"displaytime_"+str(time)+"perval_"+str(updaterate)+"updaterate_"+str(samplefrac)+"samplefrac_"+targetname+"_"+str(layersize)+"neurons_feedbackrate"+str(feedbackrate)+"_eta"+str(eta)+"_weight"+str(weight_val)+"_aver_clearerr").replace(".","p")
    fp = open(savename,"w")#"neflayer_5points_id_doublerange_morevariation","w")
    dump(layer,fp)
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


