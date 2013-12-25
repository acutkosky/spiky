
#include<crand?>
#include<list>

#include<cmath>

using namespace std;

typedef real double


class HedonisticSynaps {
  real q;
  real delta_c;
  real c;
  real tau_c;
  real tau_r;
  real tau_e;
  bool available;
  real trace_e;
  real W;
  real G;
  real V_rev;

public:

  HedonisticSynapse(real initialQ=0.0,real adelta_c=0.0,real atau_c=1.0,real atau_r=1.0,real atau_e=1.0,real atau_g=1.0,real aW=2.4,real aV_rev=0.0):
    q(initialQ),delta_c(adelta_c),tau_c(atau_c),tau_r(atau_r),tau_e(atau_r),available(astate),trace_e(atrace_d),tau_g(atau_g),W(aW),V_rev(aV_rev), c(0.0), G(0.0) {}

  bool Process(bool gotspike,real deltaT) {
    int release = 0;
    if(gotspike && available) {
      real p = 1.0/(1.0+exp(-q-c));
      real r = random();

      if(r<p) {
	//time to release!
	release = 1;
	trace_e += 1-p;
	available = 0;
      } else {
	release = 0;
	trace_e += -p;
      }
    } else {
      trace_e += -trace_e/tau_e*deltaT;
      c += -c/tau_c*deltaT;
      if(random() < deltaT/tau_r)
	available = true;
    }

    G += W*release;
    G -= G/tau_g*deltaT;
    return release;
  }

  void Update(real h_val, real eta) {
    q += eta*h_val*trace_e;
  }

  real get_G(void) {
    return G;
  }

  real get_V_rev(void) {
    return V_rev;
  }


};


class Neuron {
  real C;
  real g_L;
  real V_l;
  real I_tonic;
  real V_t;
  real V_r;
  list<HedonisticSynapse> inputs;
  list<HedonisticSynapse> outputs;
  real V;

public:

  Neuron(real aC,real ag_L, real aV_L, real aI_tonic, real aV_t, real aV_r): C(aC),g_L(ag_L),V_L(aV_L),I_tonic(aI_tonic),V_t(aV_t),V_r(aV_r),V(aV_r) {
    inputs.clear();
    outputs.clear;
  }

  int Update(real deltaT) {
    real d = -g_L*(V-V_L)+I_tonic;
    for(list<HedonisticSynapse>::iterator synapse=inputs.begin();snapse!=inputs.end();++synapse)
      d -= synapse->get_G()*(V-synapse->get_V_rev());

    V += deltaT*d/C;

    int spike = 0;
    if(V>V_t) {
      spike = 1;
      V = V_r;
    }

    for(list<HedonisticSynapse>::iterator synapse = outputs.begin();synapse!=outputs.end();synapse++)
      synapse->Process(spike,deltaT);

    return spike;

  }

};

class Poisson_Spiker {

  real rate;
  bool state;
  list<HedonisticSynapse> outputs;

public:
  Poisson_Spiker(real arate, bool astate): rate(arate),state(astate) {}

  int generate(real deltaT) {
    int ret = 0;
    if(state) {
      r = random();
      if(r<rate*deltaT) {
	ret = 1
      }
    }
    for(list<HednisticSynapse>::iterator synapse = outputs.begin();synapse != outputs.end();synapse++)
      synapse->Process(ret,deltaT);

    return ret;
  }

  void swapstate(void) {
    state = !state;
  }

  void setstate(bool astate) {
    state = astate;
  }

  real get_rate(void) {
    return rate;
  }

  bool get_state(void) {
    return state;
  }
};

