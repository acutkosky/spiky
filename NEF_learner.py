import NEF
import NEF_utilities as NEFutil
from random import random
from pickle import dump
from math import sin
import sys


def target(x=None):
    target.name = "sin"
    if (x==None):
        return

    #    return 400*sin(3.141592654*(x[0]-x[1])/400)
    
    return 400*sin(3.141592654*x/400)
target()

def error(x,t):
    return -(x-t)**2

def generator(val = None):
    fmax = 1000
    if val == None:
        val = 400*(random()*2-1.0)
    return val
    if val>0:
        fp = random()*(fmax - val)+val
        assert(fp<=fmax)
    else:
        fp = random()*(fmax+val)
        assert(fp<=fmax)
    fm = fp-val
    assert(fm<=fmax)
    assert(abs(fp-fm-val)<0.1)
    assert(fm>=0 and fp>=0 and fp<=fmax and fm<=fmax)
    return [fp,fm]#400*(random()*2-1.0)

def Train(NEFlayer,eta,regularization,traintime,averagingtime,updatetime,timeperval,deltaT,target_f,error_f,generator_f):
    t = 0
    update_t = 0
    val_t = 0
    averaging_t = 0
    while(t<traintime):
        x = generator_f()
        val_t = t
        while(t-val_t<timeperval):
            averaging_t = t
            av = 0
            while(t-averaging_t<averagingtime):
                av += NEFlayer.Process(x,deltaT)
                t += deltaT
                NEFlayer.RecordErr(0)
            xhat = av/(t-averaging_t)
            er = error_f(xhat,target_f(x))
            NEFlayer.CorrectedRecUpdate(er)
        if(t-update_t>=updatetime):
            print "updating!"
            print "time offset is :",t
            xtest = generator_f()
            print "value at x: ",xtest," is: ",NEFlayer.getaverage(xtest)," while target is: ",target_f(xtest)
            print "just trained at x: ",x," with value: ",NEFlayer.getaverage(x)," while target is: ",target_f(x)
            sys.stdout.flush()
            NEFlayer.CorrectedUpdate(eta,regularization)
            
            update_t = t

    return t #return time elapsed

def Learn(NEFlayer,eta,regularization,totaltime,traintime,averagingtime,updatetime,timeperval,deltaT,target_f,error_f,generator_f,numtest=1000,savepref = "snapshots/",extradata="None"):
    
    t = 0
    testvals = [generator_f() for k in range(numtest)]
    rmse = NEFutil.normalRMSE(NEFlayer,target_f,testvals)
    print "starting rmse: ",rmse


    sys.stdout.flush()
    while(t<totaltime):
        t += Train(NEFlayer,eta,regularization,traintime,averagingtime,updatetime,timeperval,deltaT,target_f,error_f,generator_f)
        rmse = NEFutil.normalRMSE(NEFlayer,target_f,testvals)
        savename = (savepref + "snapshot_size"+str(len(NEFlayer.layer))+"_eta"+str(eta)+"_reg"+str(regularization)+"_updatet"+str(updatetime)+"_timeper"+str(timeperval)+"_curtime"+str(t)).replace(".","p")


        print "at time: ",t,"rmse for ",target_f.name," is: ",rmse," extra comments: "+extradata
        print "saving snapshot to: ",savename
        sys.stdout.flush()
        fp = open(savename,"w")
        dump(NEFlayer,fp)
        fp.close()



eta = 0.0000003
regularization = 100
deltaT = 0.5*NEF.ms
averagingtime = 0.1
timeperval = 0.5 
updatetime = 5.0            


traintime = 120
totaltime = 12*3600


layersize = 100

print "learning with:"
print "eta: ",eta
print "regularization: ",regularization
print "averagingtime: ",averagingtime
print "timeperval: ",timeperval
print "updatetime: ",updatetime
print "traintime: ",traintime
print "totaltime: ",totaltime
print "layersize: ",layersize
print "target: ",target.name
#print "NOISE IS ON!"
sys.stdout.flush()
NEFlayer = NEFutil.createlayer(layersize,1,noise = False)

Learn(NEFlayer,eta,regularization,totaltime,traintime,averagingtime,updatetime,timeperval,deltaT,target,error,generator,numtest=1000,savepref = "snapshots/sindecode_",extradata="notnoisy-smallertimeperval")



