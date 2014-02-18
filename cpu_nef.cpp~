#ifndef CUDA_NEF_CPP
#define CUDA_NEF_CPP

#include"cuda_nef.h"
#include<cmath>
#include<random>
#include<ctime>
#include<iostream>
using namespace std;
#define DIM 1

namespace NEF {
__device__ __host__  float dotp(float *a,float *b,int d) {
    float p = 0.0;
    for(int i=0;i<d;i++)
      p += a[i]*b[i];
    return p;
  }



__device__ __host__ unsigned TausStep(unsigned &z, int S1, int S2, int S3, unsigned M)  
  {  
    unsigned b=(((z << S1) ^ z) >> S2);  
    return z = (((z & M) << S3) ^ b);  
  }  

__device__ __host__ unsigned LCGStep(unsigned &z, unsigned A, unsigned C)  
  {  
    return z=(A*z+C);  
  }  

__device__ __host__ float HybridTaus(unsigned &z1,unsigned &z2, unsigned &z3, unsigned &z4)  
  {  
    // Combined period is lcm(p1,p2,p3,p4)~ 2^121  
    return 2.3283064365387e-10 * (              // Periods  
	TausStep(z1, 13, 19, 12, 4294967294UL) ^  // p1=2^31-1  
	TausStep(z2, 2, 25, 4, 4294967288UL) ^    // p2=2^30-1  
	TausStep(z3, 3, 11, 17, 4294967280UL) ^   // p3=2^28-1  
	LCGStep(z4, 1664525, 1013904223UL)        // p4=2^32  
						);  
  }

  
__device__ __host__ int Random::flipcoin(float bias) {
    return (HybridTaus(z1,z2,z3,z4)<bias);
  }


__device__ __host__ void Randomize(Random &r,int seed) {
    std::default_random_engine generator;    
    generator.seed(seed);
    r.z1 = generator();
    r.z2 = generator();
    r.z3 = generator();
    r.z4 = generator();
  }



__device__ __host__  float Pval(float q) {
    return 1.0/(1.0+exp(-q));
  }
    

__device__ __host__  int Synapse::Process(int gotspike) {
    //no conditionals - faster to not branch on a gpu (I think)
    
    int release = randomizer.flipcoin(p);



    e_track += gotspike*(release - p);
    e_count += gotspike;
    //    if(gotspike)
    //cout<<"e_traclk: "<<e_track<<endl;

    return release*gotspike;
  }

__device__ __host__  void Synapse::RecordErr(float err) {
    if(e_count>0) {
      //cout<<"e_track: "<<e_track<<" e_count: "<<e_count<<endl;
      pert_track += (e_track/e_count);
      pertsq_track += (e_track/e_count)*(e_track/e_count);
      corr_track += (e_track/e_count)*err;
      err_track += err;

      //      cout<<"pert_track: "<<pert_track<<endl;
    }
    count++;
    e_track = 0.0;
    e_count = 0;

  }

__device__ __host__  void Synapse::Update(float eta,float regularization) {
    float avpertsq = pertsq_track/count;
    float avpert = pert_track/count;
    float avcorr = corr_track/count;
    float averr = err_track/count;
    p = Pval(q);
    if(avpertsq != avpert*avpert) {

      float estimate = (avcorr - averr*avpert)/(avpertsq-avpert*avpert);

      q += eta*(estimate - regularization *p);
      if(q<-8)
	q = -8;
      if(q>8)
	q = 8;
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
    

    
__device__ __host__  template <int d> float Neuron<d>::a(float *x) {
    if(alpha*dotp(e,x,dimension())+J_bias <= J_th)
      return 0.0;
    return 1.0/(tau_ref-tau_RC*log(1.0-J_th/(alpha*dotp(e,x,dimension())+J_bias)));
  }

  template <int d> float Neuron<d>::average(float *x) {
    Pos.p = Pval(Pos.q);
    Neg.p = Pval(Neg.q);
    return a(x)*(Pos.p-Neg.p);
  }

  template <int d> int Neuron<d>::dimension(void) {
    return d;
  }

__device__  template <int d> int Neuron<d>::Process(float *x,float delta_T) {
    float rate = a(x);
    int spike = randomizer.flipcoin(rate*delta_T);
    return Pos.Process(spike)-Neg.Process(spike);
  }

__device__  template <int d> void Neuron<d>::RecordErr(float err) {
    Pos.RecordErr(err);
    Neg.RecordErr(err);
  }

__device__  template <int d> void Neuron<d>::Update(float eta,float regularization) {
    Pos.Update(eta,regularization);
    Neg.Update(eta,regularization);
  }


  void randunit(float *e,int d,int seed) {
    std::default_random_engine generator;    
    generator.seed(seed);
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

  void SetupSynapse(Synapse &s,float mean,int seed) {
    std::default_random_engine generator;    
    generator.seed(seed);
    std::exponential_distribution<float> distribution_q(mean);

    float p = distribution_q(generator);
    if(p>0.99)
      p = 0.99;

    s.q = -log(1/p -1);
    if(s.q<-5.0)
      s.q = -5.0;
    if(s.q>5.0)
      s.q = 5.0;
       
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


    std::default_random_engine generator;    
    generator.seed(seed1);
    std::normal_distribution<float> distribution_alpha(17.0*nA,5*nA);
    std::uniform_real_distribution<float> distribution_Jbias(-13*nA,27*nA);
    std::normal_distribution<float> distribution_tauref(1.5*ms,0.3*ms);
    std::normal_distribution<float> distribution_taurc(20*ms,4*ms);
    std::normal_distribution<float> distribution_Jth(1*nA,0.2*nA);

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
    std::mt19937 generator;
    generator.seed(time(0));
    
    for(int i=0;i<size;i++) {

      layer[i] = CreateNeuron<d>(size,generator(),generator(),generator(),generator(),generator());
    }
  }


  float ProcessLayer(Neuron<DIM> *layer, int size,float *x,float delta_t,float
	       process_time) {
    int a = 0;
    for(int i=0;i<size;i++) {
      for(float t = 0;t<process_time;t+=delta_t) {
	a += layer[i].Process(x,delta_t);
      }
    }

    return a/process_time;
  }


  __global__ void d_ProcessLayer(Neuron<DIM> *layer,float *x,float
				  delta_t,float process_time,int *spikes) {
    int a = 0;
    int i = blockIdx.x*blockDim.x+threadIdx.x;
 
    for(float t = 0;t<process_time;t+=delta_t) {
      a += layer[i].Process(x,delta_t);
    }

    spikes[i] = a;


  }


  __global__ void d_SumSpikes(int *spikes,int *answer,int N) {
    //gonna sum in groups of 32 I guess
    int offset = blockIdx.x*2*blockDim.x+threadIdx.x;
    int id = threadIdx.x;

    extern __shared__ int sums[];
    sums[id] = spikes[offset]+spikes[offset+blockIdx.x];

    __syncthreads();

    for(unsigned int i=blockDim.x/2;i>32;i>>= 1) {
      if(id<i) {
	sums[id] += sums[id+i];
      }
      __syncthreads();
    }


    if(id<32) {
      sums[id] += sums[id+32];
      sums[id] += sums[id+16];
      sums[id] += sums[id+8];
      sums[id] += sums[id+4];
      sums[id] += sums[id+2];
      sums[id] += sums[id+1];
    }

    if(id == 0) {
      answers[blockIdx.x] = sums[0];
    }
  }


  float ReduceSpikes(

  void RecordErr(Neuron<DIM> *layer, int size,float err) {
    for(int i=0;i<size;i++) {
      layer[i].RecordErr(err);
    }
  }


  __global__ void d_RecordErr(Neuron<DIM> *layer, int size,float err) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    layer[i].RecordErr(err);
  }

  
  void Update(Neuron<DIM> *layer, int size, float eta, float
	      regularization) {
    for(int i=0;i<size;i++) {
      layer[i].Update(eta,regularization);
    }
  }


  
  __global__ void d_Update(Neuron<DIM> *layer, int size, float eta, float
	      regularization) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;    
    layer[i].Update(eta,regularization);
  }




  __global__ void d_SumAverages(float *averages,float *answer,int N) {

    int offset = blockIdx.x*2*blockDim.x+threadIdx.x;
    int id = threadIdx.x;

    extern __shared__ float sums[];
    sums[id] = averages[offset]+averages[offset+blockIdx.x];

    __syncthreads();

    for(unsigned int i=blockDim.x/2;i>32;i>>= 1) {
      if(id<i) {
	sums[id] += sums[id+i];
      }
      __syncthreads();
    }


    if(id<32) {
      sums[id] += sums[id+32];
      sums[id] += sums[id+16];
      sums[id] += sums[id+8];
      sums[id] += sums[id+4];
      sums[id] += sums[id+2];
      sums[id] += sums[id+1];
    }

    if(id == 0) {
      answers[blockIdx.x] = sums[0];
    }
  }
    


  __global__ void d_AverageValue(Neuron<DIM> *layer,int size, float
    *x,float *averages) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    averages[i] = layer[i].average(x);
  }


  float AverageValue(Neuron<DIM> *layer,int size, float *x) {
    float a = 0;
    for(int i=0;i<size;i++) {
      a += layer[i].average(x);
    }
    return a;
  }





    
};	     
	     
	     
#endif
