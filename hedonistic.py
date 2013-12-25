
#For the love of god make sure you stick to a constant unit convention.
#For simplicity I'm going to do mks.



from random import random
from math import exp



class Hedonistic_Synapse:
    
    def __init__(self, initialQ = 0.0, delta_c = 0.0, tau_c = 1.0, tau_r = 1.0, tau_e = 1.0, tau_g=1.0, W=2.4, V_rev=0):
        assert(tau_c > 0)
        assert(tau_r > 0)
        assert(tau_e > 0)
        assert(tau_g > 0)
        assert(delta_c >= 0)
        

        self.q = initialQ
        self.delta_c = delta_c
        self.c = 0.0
        self.tau_c = tau_c
        self.tau_r = tau_r
        self.tau_e = tau_e
        self.state = 'A'
        self.trace_e = 0.0
        self.tau_g = tau_g
        self.W = W
        self.G = 0
        self.V_rev = V_rev

    def Process(self,gotspike,deltaT):
        release = 0.0
        
        #first vescicle release or failure:
        if(gotspike and self.state == 'A'):
            p = 1.0/(1.0+exp(-self.q-self.c))
            r = random()
            
            if(r<p):
                #time to release!
                release = 1.0
                self.trace_e += 1-p
                self.state = 'R'
            else:
                #failure
                release = 0.0
                self.trace_e += -p
            self.c += self.delta_c
        else:
            self.trace_e += -self.trace_e/self.tau_e*deltaT
            self.c += -self.c/self.tau_c*deltaT
            r = random()
            if(r<deltaT/self.tau_r):
                self.state = 'A'
        self.G += self.W*release
        self.G -= self.G/self.tau_g*deltaT

        return release

    def Update(self,h_val,eta):
        self.q += eta*h_val*self.trace_e
    

    


class Neuron:

    def __init__(self,C,g_L,V_L,I_tonic,V_t,V_r,inputs = [],outputs= []):
        self.C = C
        self.g_L = g_L
        self.V_L = V_L
        self.I_tonic = I_tonic
        self.V_t= V_t
        self.V_r = V_r
        self.inputs = inputs
        self.outputs = outputs
        self.V = V_r

    def Update(self,deltaT):
        d = -self.g_L*(self.V-self.V_L)+self.I_tonic
        d -= reduce(lambda x,y:x+y,[synapse.G*(self.V-synapse.V_rev) for synapse in self.inputs])
        self.V += deltaT*d/self.C

        spike = False
        if(self.V>self.V_t):
            spike = True
            self.V = self.V_r

        for synapse in self.outputs:
            synapse.Process(spike,deltaT)
        return spike

        
class Poisson_Spiker:
    
    def __init__(self,rate,state = False,outputs = []):
        self.rate = rate
        self.state = state
        self.outputs = outputs

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

