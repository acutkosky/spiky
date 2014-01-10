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



    def Process(self,gotspike,deltaT):
        release = 0.0
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

    def Update(self,h_val,eta):
#        if(self.V_rev == 0):
#            self.q+=0.1
        self.q += eta*h_val*self.trace_e

        if(self.q > 4):
            self.q = 4
        if(self.q < -4):
            self.q = -4
    

    


class NEFneuron:
    
    def __init__(self,synapse, tau_ref = 1*ms,tau_RC = 20*ms,J_th=1*nA,J_bias = 10*nA,e=1,alpha=17*nA):
        assert(tau_ref >0)
        assert(tau_RC>0)
        self.tau_ref = tau_ref
        self.tau_RC = tau_RC
        self.J_th = J_th
        self.e = e
        self.alpha = alpha
        self.J_bias = J_bias
        self.synapse = synapse
        

    def a(self,x):

        if(self.alpha*self.e*x+self.J_bias<=0):
            return 0.0
        try:
            return 1.0/(self.tau_ref-self.tau_RC*log(1.0-self.J_th/(self.alpha*self.e*x+self.J_bias)))
        except ValueError:
            return 0.0


    def getoutput(self,x,deltaT):
        r = random()
        return (r<deltaT*self.a(x))


class NEF_layer:
    def __init__(self,layer,tau_PSC):
        #let's copy things this time instead
        self.layer = dc(layer)
        self.xhat = 0.0
        self.tau_PSC = tau_PSC
    def Process(self,x,deltaT):



        for neuron in self.layer:
            self.xhat += 0.001*neuron.synapse.inhibitory*neuron.synapse.Process(neuron.getoutput(x,deltaT),deltaT)

        self.xhat = self.xhat*exp(-deltaT/self.tau_PSC)

        return self.xhat
    def Update(self,h,eta):
        for neuron in self.layer:
            neuron.synapse.Update(h,eta)

