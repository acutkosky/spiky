

from random import random
from math import exp



class Hedonistic_Synapse:
    
    def __init__(self, initialQ = 0.0, delta_c = 0.0, tau_c = 1.0, tau_r = 1.0, tau_e = 1.0):
        assert(tau_c > 0)
        assert(tau_r > 0)
        assert(tau_e > 0)
        assert(delt_c > 0)

        self.q = initialQ
        self.delta_c = delta_c
        self.c = 0.0
        self.tau_c = tau_c
        self.tau_r = tau_r
        self.rau_e = tau_e
        self.state = A
        self.trace_e = 0.0

    def Process(self,gotspike,deltaT):
        release = false
        
        #first vescicle release or failure:
        if(gotspike):
            p = 1.0/(1.0+exp(-self.q-self.c))
            r = random()
            
            if(r<p):
                #time to release!
                release = true
                self.trace_e += 1-p
            else:
                #failure
                release = false
                self.trace_e += -p
            self.c += self.delta_c
        else:
            self.trace_e += -self.trace_e/self.tau_e*deltaT
            self.c += -self.c/self.tau_c*deltaT

        return release
