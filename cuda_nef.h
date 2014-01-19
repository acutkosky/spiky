
#include<stdlib.h>
#include<vector>
using namespae std;

//constants!
const real ms = 1.0;//0.001;
const real nA = 1.0;//0.000000001;



typdef float real;

//each synapse needs to have an in-house random number generator

class Random {
  //TODO: modify to actually do something useful on a gpu

 public:
  real operator()(void) {
    return ((real)rand())/RAND_MAX;
  }
};


class Synapse {

  char inhibitory;
  real q;
  char spiked;
  real error;
  int avcounter;
  real erupdate;
  real weight;

  Random random;

 public:

 Synapse(real a_weight,char a_inhibitory = (char)1.0,real a_q = 0.0,Random a_random):
  weight(a_weight),inhibitory(a_inhibitory),q(a_q),spiked(0),error(0),avcounter(0),erupdate(0),random(a_random) {}


  real weight(void);

  real Pval(void);

  real Process(char gotspike,real deltaT);

  void RecordErr(real erval);

  void Update(real eta);
};


class NEF_neuron {


  real tau_ref;
  real tau_RC;
  real J_th;
  real alpha;
  real J_bias;
  

  //update to multi-D later
  real e;

  //just hard-code two synapses for now...
  Synapse excite;
  Synapse inhibit;

  Random random;

 public:

 NEF_neuron(Synapse a_excite, Synapse a_inhibit, real a_tau_ref, real a_tau_RC. real a_J_th, real a_alpha, real a_J_bias, real a_e,Random a_random):
  excite(a_excite), inhibit(a_inhibit), tau_ref(a_tau_ref), tau_RC(a_tau_RC), J_th(a_J_th), alpha(a_alpha), J_bias(a_J_bias), e(a_e) random(a_random) {}
  
  real a(real x);
  
  real getoutput(real x, real deltaT);

};


//probably this struct will need a serious overhaul for the gpu
//but we'll just implement in for single-threaded cpu here first
struct NEF_layer {
  
  vector<NEF_neuron> layer;
  real tau_PSC;
  real xhat;
  real eta;

public:
  
  real Process(real x, real deltaT);

  real getaverage(real x);

  real RecordErr(real erval);

  real Update(void);



};

