
import NEF
import NEF_utilities as NEFutil
from random import random
from pickle import dump


def target(x):
    return x

def error(x,t):
    return -(x-t)**2

def generator():
    return 400*(random()*2-1.0)

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
            er = error(xhat,target(x))
            NEFlayer.CorrectedRecUpdate(er)
        if(t-update_t>=updatetime):
            NEFlayer.CorrectedUpdate(eta,regularization)
            update_t = t

    return t #return time elapsed

def Learn(NEFlayer,eta,regularization,totaltime,traintime,averagingtime,updatetime,timeperval,deltaT,target_f,error_f,generator_f,numtest=1000,savepref = "snapshots/"):
    
    t = 0
    
    testvals = [400*(random()*2.0-1.0) for k in range(numtest)]

    while(t<totaltime):
        t += Train(NEFlayer,eta,regularization,traintime,averagingtime,updatetime,timeperval,deltaT,target_f,error_f,generator_f)
        rmse = NEFutil.normalRMSE(NEFlayer,target_f,testvals)
        savename = (savepref + "snapshot_size"+str(len(NEFlayer.layer))+"_eta"+str(eta)+"_reg"+str(regularization)+"_updatet"+str(updatetime)+"_timeper"+str(timeperval)+"_curtime"+str(t)).replace(".","p")


        print "at time: ",t
        print "rmse is: ",rmse
        print "saving snapshot to: ",savename

        fp = open(savename,"w")
        dump(NEFlayer,fp)
        fp.close()


eta = 0.0001
regularization = 0.00001
deltaT = NEF.ms
averagingtime = 0.1
timeperval = 1.0 
updatetime = 10.0            


traintime = 120
totaltime = 12*3600



NEFlayer = NEFutil.createlayer(10,1)

Learn(NEFlayer,eta,regularization,totaltime,traintime,averagingtime,updatetime,timeperval,deltaT,target,error,generator,numtest=1000,savepref = "snapshots/")



