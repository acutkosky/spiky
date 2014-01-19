#include "cuda_nef.h"
#include<cmath>
using namespace std;


real Synapse::Pval(void) {
  return 1.0/(1.0+exp(-q));
}

real Synapse::Process(real gotspike) {
    
  real retval = 0.0;
  erupdate = 0.0;

  real p = Pval();
  
  r = random();

  if(gotspike>0) {
    if(r<p) {
      //released spike!
      erupdate = (1-p);
      retval = weight;
    } else {
      erupdate = (-p);
    }
  }

  return retval;

}

void Synapse::weight(void) {
  return weight;
}

void Synapse::RecordErr(real erval) {
  error += erupdate*erval;
  avcounter++;
}

void Synapse::Update(real eta) {
  q += eta*error/avcounter;
  error = 0.0;
  avcounter = 0;
}
 

real NEF_neuron::a(real x) {
  if(alpha*e*x+J_bias <= J_th):
    return 0.0;

  return 1.0/(tau_ref - tau_RC*log(1.0-J_th/(alpha*e*x+J_bias)));

}

  
real NEF_neuron::getoutput(real x, real deltaT) {
  r = random();
  return deltaT*a(x)-r;
};



//this is a major parallel procedure. Needs big overhaul for GPU
real NEF_layer::Process(real x, real deltaT) {

  real spike;
  real delta = 0;
  for(vector<NEF_neuron>::iterator it=layer.begin();it<layer.end();it++) {
    spike = it->getoutput(x,deltaT);
    delta += (it->excite.Process(spike)-it->inhibit.Process(spike));
  }
  xhat += tau_PSC*delta;
  xhat = xhat*exp(-deltaT/tau_PSC);

  return xhat;
}

real NEF_layer::getaverage(real x) {
  real av = 0;
  for(vector<NEF_neuron>::iterator it= layer.begin();it<layer.end();it++) {
    av += it->a(x)*(it->excite.Pval()*it->excite.weight()-it->inhibit.Pval()*it->inhibit.weight());
  }
  return av*tau_PSC;
}


void NEF_layer::RecordErr(real erval) {
  for(vector<NEF_neuron>::iterator it=layer.begin();it<layer.end();it++) {
    it->excite.RecordErr(erval);
    it->inhibit.RecordErr(erval);
  }
}

void NEF_layer::Update(void) {
  for(vector<NEF_neuron>::iterator it=layer.begin();it<layer.end();it++) {
    it->excite.Update(eta);
    it->inhibit.Update(eta);
  }
}
