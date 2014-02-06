from sparseNEF import SparseNEF
import NEF_utilities as Nutil
from random import random
from math import sin



def target(xval):
    return 400*(sin(3.141592654*(reduce(lambda x,y:x+y,xval,0))/(5*400.0)))

dim = 20

#Nutil.randunit(20)
#exit()
net = Nutil.createsparselayer(30*dim,30*dim,dim,dim)


print "created!"
trainsize = 1000
test = [400*(random())*Nutil.randunit(dim) for x in range(trainsize)]
targets = [target(x) for x in test]
print "maxval: ",max(targets)
print "minval: ",min(targets)

examples = [400*(random())*Nutil.randunit(dim) for x in range(trainsize)]


net.SolveEverything(examples,target,5,5)
print "solved!"



print Nutil.sparseRMSE(net,target,test)#[400*(random())*Nutil.randunit(dim) for x in range(trainsize)])
#Nutil.plotsparseavs(net,target,-400,400,1)
