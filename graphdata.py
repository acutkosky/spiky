#!/bin/python

from pickle import load
from matplotlib import pyplot as plt
from numpy import array

#fp = open("progressfile")
fp = open("testerdump_zero")
data = load(fp)

fp.close()

inputs = data[0]
l = len(inputs[0])/2
inspikes = data[2]
xs = array([reduce(lambda x,y:x+y,place[:l])/float(l) for place in inputs])
ys = array([reduce(lambda x,y:x+y,place[l:])/float(l) for place in inputs])
ps = xs+ys
#ys = [y/(2.0*max(ys)) for y in ys]
yss = [reduce(lambda x,y:x+y,place[l:])/float(l)*(10**(-2))-0.08 for place in inspikes]
xss = [reduce(lambda x,y:x+y,place[:l])/float(l)*0.01-0.075 for place in inspikes]
zs = [x for x in data[1]]
#print data[1]
#plt.plot(zs[-2000:])
#plt.plot(ys[-2000:])
#plt.plot(xs[-2000:])
plt.plot(zs)
#plt.plot([x*0.01-0.07 for x in data[3]])
#plt.plot(ps)
halfwindow = 4000
freq = []
#addendum = [0.0]*halfwindow+data[3]+[0.0]*halfwindow

#freq = [reduce(lambda x,y:x+y,addendum[i-halfwindow:i+halfwindow])/(halfwindow*0.001) for i in range(halfwindow,len(data[3])+halfwindow)]


#plt.plot(freq)

#plt.plot(ys)
#plt.plot(yss)
#plt.plot(xss)
#plt.plot(xs)
print data[4]
plt.show()
