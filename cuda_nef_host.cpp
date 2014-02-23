#ifndef CUDA_NEF_HOST_CPP
#define CUDA_NEF_HOST_CPP
#include"cuda_nef.cu.h"
#include<boost/random.hpp>
#include<ctime>
using namespace boost::random;

using namespace boost::random;

namespace NEF {



  void Randomize(Random &r,int seed) {
    boost::random::mt19937 generator;    
    generator.seed(seed);
    r.z1 = generator();
    r.z2 = generator();
    r.z3 = generator();
    r.z4 = generator();
  }


  void randunit(float *e,int d,int seed) {
    boost::random::mt19937 generator;    
    generator.seed(seed);
    boost::random::normal_distribution<float> distribution(0,2);
    
    float norm = 0;
    for(int i=0;i<d;i++) {
      float r = distribution(generator);
      norm += r*r;
      *(e+i) = r;
    }
    norm = sqrt(norm);
    for(int i=0;i<d;i++) {
      *(e+i) /= norm;
    }
  }


  void randvalue(float *e,int d,float max, int seed) {
    boost::random::mt19937 generator;    
    generator.seed(seed);
    boost::random::normal_distribution<float> distribution(0,2);
    boost::random::uniform_real_distribution<float> r_distribution(0,max);
    float norm = 0;
    for(int i=0;i<d;i++) {
      float r = distribution(generator);
      norm += r*r;
      *(e+i) = r;
    }
    norm = sqrt(norm);
    for(int i=0;i<d;i++) {
      *(e+i) *= r_distribution(generator)/norm;
    }

  }


  void SetupSynapse(Synapse &s,float mean,int seed) {
    boost::random::mt19937 generator;    
    generator.seed(seed);
    boost::random::exponential_distribution<float> distribution_q(mean);

    float p = distribution_q(generator);
    //p = 0.0;
    if(p>0.99)
      p = 0.99;

    s.q = -log(1/p -1);

    if(s.q<-10.0)
      s.q = -10.0;
    if(s.q>10.0)
      s.q = 10.0;
       
    s.p = Pval(s.q);
    

    //cout<<"pval: "<<s.p<<endl;
    //cout<<"p: "<<p<<endl;

    s.e_track=0.0;
    s.e_count = 0;
    s.pert_track = 0.0;
    s.pertsq_track = 0.0;
    s.corr_track = 0.0;
    s.err_track = 0.0;
    s.count = 0;

    Randomize(s.randomizer,generator());

  }

  template <int d> Neuron<d> CreateNeuron(float size,int seed1,int seed2 = 0,int
					  seed3 = 0,int seed4 = 0, int seed5 = 0) {

    float nA = 0.000000001;
    float ms = 0.001;


    boost::random::mt19937 generator;    
    generator.seed(seed1);
    boost::random::normal_distribution<float> distribution_alpha(17.0*nA,5*nA);
    boost::random::uniform_real_distribution<float> distribution_Jbias(-13*nA,27*nA);
    boost::random::normal_distribution<float> distribution_tauref(1.5*ms,0.3*ms);
    boost::random::normal_distribution<float> distribution_taurc(20*ms,4*ms);
    boost::random::normal_distribution<float> distribution_Jth(1*nA,0.2*nA);

    float alpha = (1.0/400.0)*distribution_alpha(generator);
    float J_bias = distribution_Jbias(generator);
    float tau_ref = distribution_tauref(generator);
    float tau_RC = distribution_taurc(generator);
    float J_th = distribution_Jth(generator);


    Neuron<d> N;

    N.alpha = alpha;
    N.J_bias = J_bias;
    N.tau_ref = tau_ref;
    N.tau_RC = tau_RC;
    N.J_th = J_th;

    randunit(N.e,d,seed2^generator());


    Randomize(N.randomizer,seed3^generator());

    SetupSynapse(N.Pos,(size+5.0),seed4^generator());
    SetupSynapse(N.Neg,(size+5.0),seed5^generator());


    return N;
  }


  template <int d> void FillLayer(Neuron<d> *layer,int size) {
    boost::random::mt19937 generator;
    generator.seed(time(0));
    
    for(int i=0;i<size;i++) {

      layer[i] = CreateNeuron<d>(size,generator(),generator(),generator(),generator(),generator());
    }
  }

};




#endif
