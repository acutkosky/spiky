from sparseNEF import SparseNEF
import NEF_utilities as Nutil
from random import random
from math import sin,sqrt
import numpy as np
import sys



def target(xval):
    return 400*sin(2*3.141592654*np.linalg.norm(xval)/400)
#    yval = [xval[i] for i in range(len(xval)-1,-1,-1)]
#    return np.dot(xval,yval)/300#xval[0]/200*reduce(lambda x,y:x+y,xval,0)/2#*200*10**(-36)#400*(sin(3.141592654*(reduce(lambda x,y:x+y,xval,0))/(5*400.0)))


def error(val,tval):
    return -abs(val-tval)**2

dim = 30

innersize = int(sys.argv[1])



print "Running with dim = ",dim," and innersize: ",innersize
print "running single layer!"
#Nutil.randunit(20)
#exit()
net = Nutil.createsparselayer(30*dim,30*dim,dim,innersize*dim)

print "created!"
trainsize = 3000
test = [400*(random())*Nutil.randunit(dim) for x in range(trainsize)]

targets = [target(x) for x in test]
maxval = max(targets)
minval = min(targets)
print "maxval: ",maxval
print "minval: ",minval

examples = [400*(random())*Nutil.randunit(dim) for x in range(trainsize)]

rmse =  Nutil.generalRMSE(net.GetInnerVal,target,test)#[400*(random())*Nutil.randunit(dim) for x in range(trainsize)])
print "starting rmse: ",rmse

#net.SolveSingleLayer(examples,target,int(argv[1]))
net.SolveOuterWeights(examples,target,100)

print "solved outer layer only, regularization 100:"

sys.stdout.flush()
rmse = Nutil.generalRMSE(net.GetVal,target,test)
print "rmse: ",rmse

sys.stdout.flush()


print "starting on full solution, regularization 10, 100"
net.SolveEverything(examples,target,10,100)
print "solved!"

sys.stdout.flush()

rmse =  Nutil.sparseRMSE(net,target,test)#[400*(random())*Nutil.randunit(dim) for x in range(trainsize)])
print "RMSE: ",rmse
print "percent: ",rmse/max(abs(maxval),abs(minval))
exit()

if len(sys.argv) == 1:
    samples = 100*dim
    trials = 100
    eta = 0.0000003
    reg = 10
else:
    samples = int(sys.argv[1])
    trials = int(sys.argv[2])
    eta = float(sys.argv[3])
    reg = float(sys.argv[4])


    
print "samples: ",samples
print "trials: ",trials
print "eta: ",eta
print "reg: ",reg
print "dim: ",dim

sys.stdout.flush()
xval = 400*Nutil.randunit(dim)
xval2 = 400*Nutil.randunit(dim)
xval3 = 400*random()*Nutil.randunit(dim)
#xval = np.array([400])
#xval2 = np.array([-400])
for i in range(trainsize):
    print "training iteration: ",i
    for j in range(samples):
#        net.TrainX(xval,target,error,500,1)
#        net.TrainX(xval2,target,error,500,1)
        net.TrainX(400*(random())*Nutil.randunit(dim),target,error,trials,1)
#        net.TrainX(400*(random())*Nutil.randunit(dim),target,error,100,.01)

#    net.UpdateGrad()
 
    net.Update(eta,reg)

    print "val: ",net.GetVal(xval)," xval: ",xval," target: ",target(xval)
    print "val: ",net.GetVal(xval2)," xval: ",xval2," target: ",target(xval2)
    print "val: ",net.GetVal(xval3)," xval: ",xval3," target: ",target(xval3)
#    print "diff: ",net.GetVal(xval,True)-net.GetVal(xval2,True)

    if(i%20==0):
        print "rmse: ",Nutil.sparseRMSE(net,target,test)
    sys.stdout.flush()

    #net.SolveEverything(examples,target,1,1)
print "solved!"



rmse =  Nutil.sparseRMSE(net,target,test)#[400*(random())*Nutil.randunit(dim) for x in range(trainsize)])
print "RMSE: ",rmse
print "percent: ",rmse/max(abs(maxval),abs(minval))
#Nutil.plotsparseavs(net,target,-400,400,1)
