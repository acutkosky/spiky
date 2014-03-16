
from NEF import NEF_layer,NEFneuron
from matplotlib import pyplot
#from NEF_utilities import *
from copy import deepcopy
from random import choice,sample,random,normalvariate,expovariate
import numpy as np
from math import log

class SparseNEF:
    
    def __init__(self,Layer1,Layer2,numconnections):
        self.Layer1 = deepcopy(Layer1)
        self.Layer2 = deepcopy(Layer2)
        self.numconnections = numconnections

        #ok,time to make connections:
        self.connectlist = []
        for i in range(len(Layer2)):
            self.connectlist.append(sample(range(len(Layer1)),numconnections))

        self.innerweights =np.array([np.array([random()*0.5-0.25 for i in range(numconnections)]) for z in Layer2])

        self.outerweights = np.array([random()*0.5-0.25 for z in Layer2])

        self.innerperturbs = np.array([np.zeros(numconnections) for z in Layer2])
        self.outerperturbs = np.array([0.0 for z in Layer2])

        self.innersquareperturbs = np.array([np.zeros(numconnections) for z in Layer2])
        self.outersquareperturbs = np.array([0.0 for z in Layer2])


        self.innercorrelations = np.array([np.zeros(numconnections) for z in Layer2])
        self.outercorrelations = np.array([0.0 for z in Layer2])

        self.innergrad = np.array([np.zeros(numconnections) for z in Layer2])
        self.outergrad = np.array([0.0 for z in Layer2])
        self.count = 0

        self.gradcount = 0

        self.AverageEr = 0

    def NormalTwoLayer(self):
        for i in range(len(self.Layer2)):
            self.connectlist[i] = range(len(self.Layer1))
        self.innerweights =np.array([np.array([random()*0.5-0.25 for i in range(self.numconnections)]) for z in self.Layer2])
        

    def OuterVals(self,x):
        firstlayer = [neuron.a(x) for neuron in self.Layer1]
        
        secondlayer = []

        for i in range(len(self.Layer2)):
            t = 0
            for j in range(self.numconnections):
                t += firstlayer[self.connectlist[i][j]]*self.innerweights[i][j]
            secondlayer.append(self.Layer2[i].a(t))

        return secondlayer

    def GetInnerVal(self,x):
        firstlayer = [neuron.a(x) for neuron in self.Layer1]
        retval = 0
        for i in range(len(self.Layer2)):
            t = 0
            for j in range(self.numconnections):
                t += firstlayer[self.connectlist[i][j]]*self.innerweights[i][j]
            retval += t
        return retval/len(self.Layer2)

    def GetVal(self,x,foo = False):

        firstlayer = [neuron.a(x) for neuron in self.Layer1]
        
        secondlayer = []
        ret_val = 0
        for i in range(len(self.Layer2)):
            t = 0
            for j in range(self.numconnections):
                t += firstlayer[self.connectlist[i][j]]*self.innerweights[i][j]
            secondlayer.append(self.Layer2[i].a(t))
            ret_val += t
        
        val = np.dot(self.outerweights,secondlayer)
#        return 0
        if(foo):
            return val
        return val

    def TrainX(self,xval,target,errorf,iterations,alpha):
        for r in range(iterations):
            curinnerperturbs = np.array([np.array([alpha*(random()*2-1) for x in range(self.numconnections)]) for z in self.Layer2])
            curouterperturbs = np.array([alpha*(random()*2-1) for z in self.Layer2])
            
            self.innerperturbs += curinnerperturbs
            self.outerperturbs += curouterperturbs

            self.innersquareperturbs += curinnerperturbs**2
            self.outersquareperturbs += curouterperturbs**2


            self.count += 1
            
            self.innerweights += curinnerperturbs
            self.outerweights += curouterperturbs



            
            val = self.GetVal(xval)
            
            self.innerweights -= curinnerperturbs
            self.outerweights -= curouterperturbs
            
            Er = errorf(target(xval),val)

            self.AverageEr += Er
            
            self.innercorrelations += Er*curinnerperturbs
            self.outercorrelations += Er*curouterperturbs
    
    def UpdateGrad(self):
        innergradbias = self.innercorrelations/self.count - self.innerperturbs/self.count*self.AverageEr/self.count

        innergradest = innergradbias/(self.innersquareperturbs/self.count - (self.innerperturbs/self.count)**2)

        outergradbias = self.outercorrelations/self.count - self.outerperturbs/self.count*self.AverageEr/self.count

        outergradest = outergradbias/(self.outersquareperturbs/self.count - (self.outerperturbs/self.count)**2)



        self.innerperturbs = np.array([np.zeros(self.numconnections) for z in self.Layer2])
        self.outerperturbs = np.array([0.0 for z in self.Layer2])

        self.innersquareperturbs = np.array([np.zeros(self.numconnections) for z in self.Layer2])
        self.outersquareperturbs = np.array([0.0 for z in self.Layer2])


        self.innercorrelations = np.array([np.zeros(self.numconnections) for z in self.Layer2])
        self.outercorrelations = np.array([0.0 for z in self.Layer2])

        self.count = 0

        self.AverageEr = 0

        self.gradcount += 1

        self.innergrad += innergradest
        self.outergrad += outergradest

    def Update(self,alpha,regularization):
        innergradbias = self.innercorrelations/self.count - self.innerperturbs/self.count*self.AverageEr/self.count

        innergradest = innergradbias/(self.innersquareperturbs/self.count - (self.innerperturbs/self.count)**2)

        outergradbias = self.outercorrelations/self.count - self.outerperturbs/self.count*self.AverageEr/self.count

        outergradest = outergradbias/(self.outersquareperturbs/self.count - (self.outerperturbs/self.count)**2)


        print "max: ",innergradest[0][0]
        print "sq: ",self.innersquareperturbs[0][0]/self.count
        print "nonsq: ",self.innerperturbs[0][0]/self.count
        print "weight: ",self.innerweights[0][0]

        self.innerperturbs = np.array([np.zeros(self.numconnections) for z in self.Layer2])
        self.outerperturbs = np.array([0.0 for z in self.Layer2])

        self.innersquareperturbs = np.array([np.zeros(self.numconnections) for z in self.Layer2])
        self.outersquareperturbs = np.array([0.0 for z in self.Layer2])


        self.innercorrelations = np.array([np.zeros(self.numconnections) for z in self.Layer2])
        self.outercorrelations = np.array([0.0 for z in self.Layer2])

        self.count = 0

        self.AverageEr = 0

        self.innerweights += alpha*innergradest-alpha*regularization*self.innerweights

        self.outerweights += alpha*outergradest-alpha*regularization*self.outerweights
        
        self.gradcount = 0
        self.innergrad = np.array([np.zeros(self.numconnections) for z in self.Layer2])
        self.outergrad = np.array([0.0 for z in self.Layer2])        

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
#        print np.shape(T)
        Gamma = M.transpose()*T

        ID = regularization*np.identity(np.shape(Lambda)[0])

        d = (Lambda +ID*ID)**(-1)*Gamma
        
        dlist = list(np.array(d.transpose())[0])

  #      for x in dlist:
  #          assert(x<1 and x>-1)

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

   #     for x in dlist:
   #         assert(x<1 and x>-1)

        self.outerweights = dlist

        return self.outerweights

    def SolveSingleLayer(self,xvals,target,reg=10):
        for i in range(len(self.Layer2)):
            self.SolveX(xvals,i,target,reg)


    def GetSVDrate(self,xvals,dim):

        indices = range(len(self.Layer1))


        M = np.matrix([[self.Layer1[j].a(x) for j in indices] for x in xvals])


        Lambda = M.transpose()*M

        eigs = [log(x) for x in list(np.linalg.eig(Lambda)[0]) if abs(x)> 0.01]
        
        eigs.sort()
        eigs.reverse()


        
        data = np.polyfit(range(len(eigs)),eigs,1)
        slope = data[0]
        intercept = data[1]
        print "slope: ",slope
        print "intercept: ",intercept
        pyplot.clf()
        pyplot.plot(range(len(eigs)),eigs)
        pyplot.plot(range(len(eigs)),[slope*x+intercept for x in range(len(eigs))])
        pyplot.title("num neurons: "+str(len(self.Layer1))+" slope: "+str(slope))
        pyplot.savefig("dim"+str(dim))
        #pyplot.show()
        return slope



    

    def SolveNormalTwo(self,xvals,target,reginner,regouter):
        self.SolveX(xvals,0,target,reginner)
        for i in range(1,len(self.Layer2)):
            self.innerweights[i] = self.innerweights[0]
        print "Solved inner layer"
        self.SolveOuterWeights(xvals,target,regouter)

        
    def SolveEverything(self,xvals,target,reginner=10,regouter=100):
        
        for i in range(len(self.Layer2)):
            self.SolveX(xvals,i,target,reginner)
        print "Solved inner layer"
        self.SolveOuterWeights(xvals,target,regouter)

