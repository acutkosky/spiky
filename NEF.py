from math import log,exp
from copy import deepcopy as dc
from random import random
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
        return 1.0/(1.0+exp(-self.q-self.c))

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
        self.q += eta*self.etrack/self.avcounter
        self.etrack = 0
        self.avcounter = 0
        


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

        if(self.alpha*self.e*x+self.J_bias<=self.J_th):
            return 0.0
        try:
            return 1.0/(self.tau_ref-self.tau_RC*log(1.0-self.J_th/(self.alpha*self.e*x+self.J_bias)))
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
        self.weight = 1
    def Process(self,x,deltaT):


        delta = 0
        av = 0
        for neuron in self.layer:
            #0.001
            spike = neuron.getoutput(x,deltaT)
            delta += reduce(lambda x,y:x+y,[self.weight*self.tau_PSC*synapse.inhibitory*synapse.Process(spike,deltaT) for synapse in neuron.synapses])
            av += reduce(lambda x,y:x+y,[self.weight*self.tau_PSC*synapse.inhibitory*synapse.Pval()*neuron.a(x) for synapse in neuron.synapses])
        self.xhat += delta
        self.average = av*self.tau_PSC#deltaT*exp(-deltaT/self.tau_PSC)/(1-exp(-deltaT/self.tau_PSC))
        self.xhat = self.xhat*exp(-deltaT/self.tau_PSC)

        return self.xhat
    def getaverage(self,x):
        av = 0 
        for neuron in self.layer:
            av += reduce(lambda x,y:x+y,[self.weight*self.tau_PSC*synapse.inhibitory*synapse.Pval()*neuron.a(x) for synapse in neuron.synapses])
        return av*self.tau_PSC

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

