from sparseNEF import SparseNEF
import NEF_utilities as Nutil
from random import random
from math import sin,sqrt
import numpy as np

def target(xval):
    return np.linalg.norm(xval)
#    yval = [xval[i] for i in range(len(xval)-1,-1,-1)]
#    return np.dot(xval,yval)/300#xval[0]/200*reduce(lambda x,y:x+y,xval,0)/2#*200*10**(-36)#400*(sin(3.141592654*(reduce(lambda x,y:x+y,xval,0))/(5*400.0)))


def error(val,tval):
    return -abs(val-tval)**2

dim = 3

#Nutil.randunit(20)
#exit()
net = Nutil.createsparselayer(30*dim,30*dim,dim,dim)


print "created!"
trainsize = 3000
test = [400*(random())*Nutil.randunit(dim) for x in range(trainsize)]
targets = [target(x) for x in test]
maxval = max(targets)
minval = min(targets)
print "maxval: ",maxval
print "minval: ",minval

examples = [400*(random())*Nutil.randunit(dim) for x in range(trainsize)]

rmse =  Nutil.sparseRMSE(net,target,test)#[400*(random())*Nutil.randunit(dim) for x in range(trainsize)])
print "starting rmse: ",rmse
xval = 400*Nutil.randunit(dim)
xval2 = 400*Nutil.randunit(dim)
xval3 = 400*random()*Nutil.randunit(dim)
#xval = np.array([400])
#xval2 = np.array([-400])
for i in range(trainsize/10):
    print "training iteration: ",i
    for j in range(50*dim):
#        net.TrainX(xval,target,error,500,1)
#        net.TrainX(xval2,target,error,500,1)
        net.TrainX(400*(random())*Nutil.randunit(dim),target,error,50,1)
#        net.TrainX(400*(random())*Nutil.randunit(dim),target,error,100,.01)

#    net.UpdateGrad()
 
    net.Update(0.0000003,10.0)#.00000001,100)#.000001,100)#.0000001,10000)#.00000001)

    print "val: ",net.GetVal(xval)," xval: ",xval," target: ",target(xval)
    print "val: ",net.GetVal(xval2)," xval: ",xval2," target: ",target(xval2)
    print "val: ",net.GetVal(xval3)," xval: ",xval3," target: ",target(xval3)
#    print "diff: ",net.GetVal(xval,True)-net.GetVal(xval2,True)

    if(i%20==0):
        print "rmse: ",Nutil.sparseRMSE(net,target,test)


#net.SolveEverything(examples,target,1,1)
print "solved!"



rmse =  Nutil.sparseRMSE(net,target,test)#[400*(random())*Nutil.randunit(dim) for x in range(trainsize)])
print "RMSE: ",rmse
print "percent: ",rmse/max(abs(maxval),abs(minval))
#Nutil.plotsparseavs(net,target,-400,400,1)
