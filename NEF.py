from math import log,exp
from copy import deepcopy as dc
from random import random
import numpy as np
mV = 0.001
pF = 10.0**(-12)
nS = 10.0**(-9)
pA = 10.0**(-12)
nA = 10.0**(-9)
ms = 0.001

class Synapse:

    
    def __init__(self, inhibitory = 1.0,initialQ = 0.0, delta_c = 0.0, tau_c = 1.0, tau_r = 0.00000001, tau_e = 1.0*ms):
        assert(tau_c > 0)
        assert(tau_r > 0)
        assert(tau_e > 0)
        assert(delta_c >= 0)
        assert(inhibitory == 1 or inhibitory == -1)

        self.inhibitory = inhibitory
        self.q = dc(initialQ)
        self.delta_c = dc(delta_c)
        self.c = 0.0
        self.tau_c = dc(tau_c)
        self.tau_r = dc(tau_r)
        self.tau_e = dc(tau_e)
        self.state = 'A'
        self.trace_e = 0.0
        self.spiked = False
        self.etrack = 0
        self.avcounter = 0
        self.processed = False

    def Pval(self):
        try:
            return 1.0/(1.0+exp(-self.q-self.c))
        except:
            print "q: ",self.q
            print "c: ",self.c
            print "fail fail"
            exit()

    def SetQfromP(self,p):
        try:
            assert(p<1)
        except:
            print "p val: ",p
            print "toooo big!!!"
            exit()

        assert(p>0)
        self.q = -log(1/float(p)-1.0)

    def Process(self,gotspike,deltaT):
        release = 0.0
        self.processed = gotspike
        self.spiked =False
        #first vescicle release or failure:
        assert(self.state == 'A')
        if(gotspike and self.state == 'A'):
            p = 1.0/(1.0+exp(-self.q-self.c))
            r = random()
#            print "gotspike"
#            assert(self.c == 0)
            if(r<p):
#                print "released"
                #time to release!
                release = 1.0
                self.spiked = True
                self.trace_e += 1-p
                self.state = 'R'
            else:
                #failure

                release = 0.0
                self.trace_e += -p
            self.c += self.delta_c
        else:
            self.trace_e = self.trace_e*exp(-deltaT/self.tau_e)
            self.c =self.c*exp( -deltaT/self.tau_c)

        r = random()
        if(r<deltaT/self.tau_r):
            self.state = 'A'




        return release


    def RecordErr(self,erval):
        p = self.Pval()
        if(self.processed):
            if(self.spiked):
                self.etrack += (1-p)*erval
            else:
                self.etrack += (-p)*erval
        self.processed = False
        self.avcounter += 1

    def RecUpdate(self,eta):
#        print self.etrack/self.avcounter
        avgrad = self.etrack/self.avcounter
        p = self.Pval()
        reggrad = -200*p*p*(1-p)
#        print "avgrad: ",avgrad
#        print "reggrad: ",reggrad
        self.q += eta*(avgrad+reggrad)
        self.etrack = 0
        self.avcounter = 0

        if(self.q>100):
            self.q = 100

        if(self.q<-100):
            self.q = -100
        


    def Update(self,h_val,eta):

        self.q += eta*h_val*self.trace_e
        self.trace_e = 0

        if(self.q > 4):
            self.q = 4
        if(self.q < -4):
            self.q = -4
    
    


class NEFneuron:
    
    def __init__(self,synapses, tau_ref = 1*ms,tau_RC = 20*ms,J_th=1*nA,J_bias = 10*nA,e=1,alpha=17*nA):
        assert(tau_ref >0)
        assert(tau_RC>0)
        self.tau_ref = tau_ref
        self.tau_RC = tau_RC
        self.J_th = J_th
        self.e = e
        self.alpha = alpha
        self.J_bias = J_bias
        self.synapses = dc(synapses)
        

    def a(self,x):

        if(self.alpha*np.dot(self.e,x)+self.J_bias<=self.J_th):
            return 0.0
        try:
            return 1.0/(self.tau_ref-self.tau_RC*log(1.0-self.J_th/(self.alpha*np.dot(self.e,x)+self.J_bias)))
        except ValueError:
            print "lamesauce"
            exit()
            return 0.0


    def getoutput(self,x,deltaT):
        r = random()
        assert(self.a(x)*deltaT<0.8)
        if (r<deltaT*self.a(x)):
            return True
        return False


class NEF_layer:
    def __init__(self,layer,tau_PSC,weight=1):
        #let's copy things this time instead
        self.layer = dc(layer)
        self.xhat = 0.0
        self.tau_PSC = tau_PSC
        self.average = 0
        self.weight = weight
    def Process(self,x,deltaT):


        delta = 0
        av = 0
        for neuron in self.layer:
            #0.001
            spike = neuron.getoutput(x,deltaT)
            delta += reduce(lambda x,y:x+y,[self.weight/self.tau_PSC*synapse.inhibitory*synapse.Process(spike,deltaT) for synapse in neuron.synapses])
            av += reduce(lambda x,y:x+y,[self.weight/self.tau_PSC*synapse.inhibitory*synapse.Pval()*neuron.a(x) for synapse in neuron.synapses])
        self.xhat += delta
        self.average = av*self.tau_PSC#deltaT*exp(-deltaT/self.tau_PSC)/(1-exp(-deltaT/self.tau_PSC))
        self.xhat = self.xhat*exp(-deltaT/self.tau_PSC)

        return self.xhat
    def getaverage(self,x):
        av = 0 
        for neuron in self.layer:
            av += reduce(lambda x,y:x+y,[self.weight*synapse.inhibitory*synapse.Pval()*neuron.a(x) for synapse in neuron.synapses])
        return av

    def getCM(self,x):
        av = 0
        for neuron in self.layer:
            av += reduce(lambda x,y:x+y,[self.weight*synapse.Pval()*neuron.a(x) for synapse in neuron.synapses])
        return av
    def RecordErr(self,erval):
        for neuron in self.layer:
            for synapse in neuron.synapses:
                synapse.RecordErr(erval)


    def RecUpdate(self,eta):
        for neuron in self.layer:
            for synapse in neuron.synapses:
                synapse.RecUpdate(eta)

    def Update(self,h,eta):
        for neuron in self.layer:
            for synapse in neuron.synapses:
                synapse.Update(h,eta)



def LeastSquaresSolve(xvals,f,neflayer):
    print "layersize: ",len(neflayer.layer)

    M = np.matrix([[neflayer.layer[i].a(xvals[j]) for i in range(len(neflayer.layer))] for j in range(len(xvals))])
    print "rank: ",np.linalg.matrix_rank(M)
    print "shape: ",np.shape(M)

    X = np.matrix([f(xvals[i]) for i in range(len(xvals))]).transpose()

#    Gamma = np.zeros(len(neflayer.layer))


#    for j in range(len(Gamma)):
#        Gamma[j] = reduce(lambda a,x:a+neflayer.layer[j].a(x)*f(x),xvals,0)

#    Gamma = np.matrix(Gamma).transpose()

#    Lambda = np.matrix( np.empty( (len(neflayer.layer),len(neflayer.layer)) ) )
#    print Lambda
#    for i in range(len(neflayer.layer)):
#        for j in range(len(neflayer.layer)):
#            Lambda[i,j] = reduce(lambda a,x:a+neflayer.layer[i].a(x)*neflayer.layer[j].a(x),xvals,0)


    Gamma = M.transpose()*X

    Lambda = M.transpose()*M
    print Lambda
    ID = 1009000*np.identity(np.shape(Lambda)[0])

    d = ((Lambda+ID)**(-1))*Gamma

    print "norm: ",np.linalg.norm(d)

    for i in range(len(neflayer.layer)):
        assert(neflayer.layer[i].synapses[0].inhibitory == 1)
        assert(neflayer.layer[i].synapses[1].inhibitory == -1)

        if d[i]>0:
            neflayer.layer[i].synapses[0].SetQfromP(d[i]/neflayer.weight)
            neflayer.layer[i].synapses[1].q = -100

        if d[i]<0:
            neflayer.layer[i].synapses[1].SetQfromP(-d[i]/neflayer.weight)
            neflayer.layer[i].synapses[0].q = -100

        if d[i] == 0:
            neflayer.layer[i].synapses[0].q = -100
            neflayer.layer[i].synapses[1].q = -100
