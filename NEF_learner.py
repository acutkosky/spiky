
import NEF
import NEF_utilities as NEFutil



def target(x):
    return x

NEFlayer = NEFutil.createlayer(100,1)
ac = 0.0
#for i in range(10):
#    for j in range(1000):
#        ac += NEFlayer.Process(0,1*NEF.ms)

#print "average: ",ac/10
print "ideal: ",NEFlayer.getaverage(0)
