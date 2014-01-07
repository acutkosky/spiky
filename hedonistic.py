
#For the love of god make sure you stick to a constant unit convention.
#For simplicity I'm going to do mks.



from random import random
from math import exp
from sys import exit
from copy import deepcopy as dc
 
class Hedonistic_Synapse:
    
    def __init__(self, initialQ = 0.0, delta_c = 0.0, tau_c = 1.0, tau_r = 1.0, tau_e = 1.0, tau_g=1.0, W=2.4, V_rev=0):
        assert(tau_c > 0)
        assert(tau_r > 0)
        assert(tau_e > 0)
        assert(tau_g > 0)
        assert(delta_c >= 0)
        

        self.q = dc(initialQ)
        self.delta_c = dc(delta_c)
        self.c = 0.0
        self.tau_c = dc(tau_c)
        self.tau_r = dc(tau_r)
        self.tau_e = dc(tau_e)
        self.state = 'A'
        self.trace_e = 0.0
        self.tau_g = dc(tau_g)
        self.W = dc(W)
        self.G = 0.0
        self.V_rev = dc(V_rev)
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
                self.G += self.W
#                self.G = self.G*exp(-deltaT/self.tau_g)
            else:
                #failure

                self.G = self.G*exp(-deltaT/self.tau_g)
                release = 0.0
                self.trace_e += -p
            self.c += self.delta_c
        else:
            self.trace_e = self.trace_e*exp(-deltaT/self.tau_e)
            self.c =self.c*exp( -deltaT/self.tau_c)

            self.G = self.G*exp(-deltaT/self.tau_g)
        r = random()
        if(r<deltaT/self.tau_r):
            self.state = 'A'




        assert(self.G>=0)
        return release

    def Update(self,h_val,eta):
#        if(self.V_rev == 0):
#            self.q+=0.1
        self.q += eta*h_val*self.trace_e

        if(self.q > 4):
            self.q = 4
        if(self.q < -4):
            self.q = -4
    

    


class Neuron:

    def __init__(self,C,g_L,V_L,I_tonic,V_t,V_r,inputs = [],outputs = []):
        self.C = dc(C)
        self.g_L = dc(g_L)
        self.V_L = dc(V_L)
        self.I_tonic = dc(I_tonic)#450*10**(-12)#dc(I_tonic)
        self.V_t= dc(V_t)
        self.V_r = dc(V_r)
        #single-depth copy!
        self.inputs = [i for i in inputs]#inputs
        self.outputs = [o for o in outputs]#outputs
        self.V = dc(V_r)
        
    def Update(self,deltaT):
        
        A = (self.g_L+reduce(lambda x,y:x+y,[synapse.G for synapse in self.inputs],0))/self.C
        N = (-self.g_L*(-self.V_L)+self.I_tonic - reduce(lambda x,y:x+y,[synapse.G*(-synapse.V_rev) for synapse in self.inputs],0))/self.C
        
        d = -self.g_L*(self.V-self.V_L)+self.I_tonic
        e = reduce(lambda x,y:x+y,[synapse.G*(self.V-synapse.V_rev) for synapse in self.inputs],0)
        d -= e
        oldV = self.V
        delta = d/self.C

        #self.V += delta*deltaT
        self.V = exp(-A*deltaT)*self.V+ (1.0/A) * (1-exp(-A*deltaT))* N
        if(self.V < -0.09):
            print "wtf"
            print "A: ",A
            print "N: ",N
            print "oldV: ",oldV
            print "delta: ",delta
            print "V: ",self.V
            print "Gs: ",[synapse.G for synapse in self.inputs]
            print "vrev: ",[synapse.V_rev for synapse in self.inputs]
            print "d: ",d
            print "gL: ",self.g_L
            print "v_L: ",self.V_L
            print "Itonic: ",self.I_tonic
            print "equilibrium (current): ",N/A
            print "equilibrium (rest): ",(-self.g_L*(-self.V_L)+self.I_tonic)/self.g_L
            exit()
#        print "d: ",d,"update: ",deltaT*d/self.C
        spike = False
        if(self.V>self.V_t):
#            print "someone spiked!"
            spike = True
            self.V = self.V_r

        for synapse in self.outputs:
            synapse.Process(spike,deltaT)
        return spike

        
class Poisson_Spiker:
    
    def __init__(self,rate,state = False,outputs = []):
        self.rate = rate
        self.state = False
        self.outputs = [o for o in outputs]

    def generate(self,deltaT):
        if(self.state == True):
            r = random()
            if(r<self.rate*deltaT):

                for synapse in self.outputs:
                    synapse.Process(True,deltaT)
                return True

        for synapse in self.outputs:
            synapse.Process(False,deltaT)
        return False
    def swapstate(self):
        self.state = not self.state
    def setstate(self,state):
        self.state = state

