
from NEF import NEF_layer,NEFneuron
#from NEF_utilities import *
from copy import deepcopy
from random import choice,sample,random,normalvariate,expovariate
import numpy as np


class SparseNEF:
    
    def __init__(self,Layer1,Layer2,numconnections):
        self.Layer1 = deepcopy(Layer1)
        self.Layer2 = deepcopy(Layer2)
        self.numconnections = numconnections

        #ok,time to make connections:
        self.connectlist = []
        for i in range(len(Layer2)):
            self.connectlist.append(sample(range(len(Layer1)),numconnections))

        self.innerweights =[np.zeros(numconnections) for z in Layer2]

        self.outerweights = [0.0 for z in Layer2]


    def OuterVals(self,x):
        firstlayer = [neuron.a(x) for neuron in self.Layer1]
        
        secondlayer = []

        for i in range(len(self.Layer2)):
            t = 0
            for j in range(self.numconnections):
                t += firstlayer[self.connectlist[i][j]]*self.innerweights[i][j]
            secondlayer.append(self.Layer2[i].a(t))

        return secondlayer


    def GetVal(self,x):
        firstlayer = [neuron.a(x) for neuron in self.Layer1]
        
        secondlayer = []

        for i in range(len(self.Layer2)):
            t = 0
            for j in range(self.numconnections):
                t += firstlayer[self.connectlist[i][j]]*self.innerweights[i][j]
            secondlayer.append(self.Layer2[i].a(t))
        
        val = np.dot(self.outerweights,secondlayer)

        return val


    def getpair(self,x):
        firstlayer = [neuron.a(x) for neuron in self.Layer1]
        
        secondlayer = []

        for i in range(len(self.Layer2)):
            t = 0
            for j in range(self.numconnections):
                t += firstlayer[self.connectlist[i][j]]*self.innerweights[i][j]
            secondlayer.append(self.Layer2[i].a(t))

        pos = 0
        neg = 0

        for i in range(len(secondlayer)):
            d = self.outerweights[i]*secondlayer[i]

            if(d>0):
                pos += d
            else:
                neg += d
        

        return [pos,neg]




    def SolveX(self,xvals,i,target,regularization = 10):

        indices = self.connectlist[i]


        M = np.matrix([[self.Layer1[j].a(x) for j in indices] for x in xvals])

        T = np.matrix([target(x) for x in xvals]).transpose()

        Lambda = M.transpose()*M

        Gamma = M.transpose()*T

        ID = regularization*np.identity(np.shape(Lambda)[0])

        d = (Lambda +ID*ID)**(-1)*Gamma
        
        dlist = list(np.array(d.transpose())[0])

#        for x in dlist:
#            assert(x<1 and x>-1)

        self.innerweights[i] = dlist


    def SolveOuterWeights(self,xvals,target,regularization = 100):

        M = np.matrix( [self.OuterVals(x) for x in xvals])

        T = np.matrix([target(x) for x in xvals]).transpose()

        print np.shape(M)


        Lambda = M.transpose()*M

        Gamma = M.transpose()*T

        ID = regularization*np.identity(np.shape(Lambda)[0])

        d = (Lambda + ID*ID)**(-1)*Gamma

        dlist = list(np.array(d.transpose())[0])

#        for x in dlist:
#            assert(x<1 and x>-1)

        self.outerweights = dlist

        return self.outerweights

    def SolveEverything(self,xvals,target,reginner=10,regouter=100):
        
        for i in range(len(self.Layer2)):
            self.SolveX(xvals,i,target,reginner)
        print "Solved inner layer"
        self.SolveOuterWeights(xvals,target,regouter)

