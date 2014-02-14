#include"cuda_nef.h"
#include<cmath>
#include<random>


namespace NEF {
  float dotp(float *a,float *b,int d) {
    float p = 0.0;
    for(int i=0;i<d;i++)
      p += a[i]*b[i];
    return p;
  }



  unsigned TausStep(unsigned &z, int S1, int S2, int S3, unsigned M)  
  {  
    unsigned b=(((z << S1) ^ z) >> S2);  
    return z = (((z & M) << S3) ^ b);  
  }  

  unsigned LCGStep(unsigned &z, unsigned A, unsigned C)  
  {  
    return z=(A*z+C);  
  }  

  float HybridTaus(unsigned &z1,unsigned &z2, unsigned &z3, unsigned &z4)  
  {  
    // Combined period is lcm(p1,p2,p3,p4)~ 2^121  
    return 2.3283064365387e-10 * (              // Periods  
	TausStep(z1, 13, 19, 12, 4294967294UL) ^  // p1=2^31-1  
	TausStep(z2, 2, 25, 4, 4294967288UL) ^    // p2=2^30-1  
	TausStep(z3, 3, 11, 17, 4294967280UL) ^   // p3=2^28-1  
	LCGStep(z4, 1664525, 1013904223UL)        // p4=2^32  
						);  
  }

  
  int Random::flipcoin(float bias) {
    return (HybridTaus(z1,z2,z3,z4)<bias);
  }


  void Randomize(Random &r) {
    std::default_random_engine generator;    
    r.z1 = generator();
    r.z2 = generator();
    r.z3 = generator();
    r.z4 = generator();
  }



  float Pval(float q) {
    return 1.0/(1.0+exp(-q));
  }
    

  int Synapse::Process(int gotspike) {
    //no conditionals - faster to not branch on a gpu (I think)
    
    int release = randomizer.flipcoin(p);

    e_track += gotspike*(release - p);
    e_count += gotspike;
    
    return release*gotspike;
  }

  void Synapse::RecordErr(float err) {
    pert_track += (e_track/e_count);
    pertsq_track += (e_track/e_count)*(e_track/e_count);
    corr_track += (e_track/e_count)*err;
    err_track += err;
    count++;
    e_track = 0.0;
    e_count = 0;
  }

  void Synapse::Update(float eta,float regularization) {
    float avpertsq = pertsq_track/count;
    float avpert = pert_track/count;
    float avcorr = corr_track/count;
    float averr = err_track/count;

    if(avpertsq != avpert*avpert) {
      float estimate = (avcorr - averr*avpert)/(avpertsq-avpert*avpert);
      q += eta*(estimate -regularization *p);
    }
    pert_track = 0;
    pertsq_track = 0;
    corr_track = 0;
    err_track = 0;
    count = 0;
    e_track = 0;
    e_count = 0;

    p = Pval(q);
  }
    

    
  template <int d> float Neuron<d>::a(float *x) {
    if(alpha*dotp(e,x,dimension())+J_bias <= J_th)
      return 0.0;
    return 1.0/(tau_ref-tau_RC*log(1.0-J_th/(alpha*dotp(e,x,dimension())+J_bias)));
  }

  template <int d> float Neuron<d>::average(float *x) {
    return a(x)*weight;
  }

  template <int d> int Neuron<d>::dimension(void) {
    return d;
  }

  template <int d> int Neuron<d>::Process(float *x,float delta_T) {
    float rate = a(x);
    int spike = randomizer.flipcoin(rate*delta_T);
    return Pos.Process(spike)-Neg.Process(spike);
  }

  template <int d> void Neuron<d>::RecordErr(float err) {
    Pos.RecordErr(err);
    Neg.RecordErr(err);
  }

  template <int d> void Neuron<d>::Update(float eta,float regularization) {
    Pos.Update(eta,regularization);
    Neg.Update(eta,regularization);
  }


  void randunit(float *e,int d) {
    std::default_random_engine generator;    
    std::normal_distribution<float> distribution(0,2);
    
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

  void SetupSynapse(Synapse &s,float mean) {
    std::default_random_engine generator;    
    std::exponential_distribution<float> distribution_q(mean);

    float q = distribution_q(generator);

    s.q = -log(1/q -1);
    s.p = Pval(q);

    s.e_track=0.0;
    s.e_count = 0;
    s.pert_track = 0.0;
    s.pertsq_track = 0.0;
    s.corr_track = 0.0;
    s.err_track = 0.0;
    s.count = 0;

    Randomize(s.randomizer);

  }

  template <int d> Neuron<d> CreateNeuron(void) {

    float nA = 0.000000001;
    float ms = 0.001;


    std::default_random_engine generator;    
    std::normal_distribution<float> distribution_alpha(17.0*nA,5*nA);
    std::uniform_real_distribution<float> distribution_Jbias(-13*nA,27*nA);
    std::normal_distribution<float> distribution_tauref(1.5*ms,0.3*ms);
    std::normal_distribution<float> distribution_taurc(20*ms,4*ms);
    std::normal_distribution<float> distribution_Jth(1*nA,0.2*nA);

    float alpha = distribution_alpha(generator);
    float J_bias = distribution_Jbias(generator);
    float tau_ref = distribution_tauref(generator);
    float tau_rc = distribution_taurc(generator);
    float J_th = distribution_Jth(generator);


    Neuron<d> N;

    N.alpha = alpha;
    N.J_bias = J_bias;
    N.tau_ref = tau_ref;
    N.tau_rc = tau_rc;
    N.J_th = J_th;

    randunit(N.e,d);


    std::normal_distribution<float> weight(0,1);
    N.weight = weight(generator);


    Randomize(N.randomizer);

    SetupSynapse(N.Pos,-1.0/d);
    SetupSynapse(N.Neg,-1.0/d);


    return N;
  }

};	     
	     
	     
