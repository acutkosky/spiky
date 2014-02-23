#ifndef CUDA_NEF_CPP
#define CUDA_NEF_CPP

#include"cpu_nef.h"
#include<cmath>
#include<random>
#include<ctime>
#include<iostream>
using namespace std;
#define DIM 1

#ifndef CUDA
#define __device__
#define __host__
#define __global__
#endif

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


  void Randomize(Random &r,int seed) {
    std::default_random_engine generator;    
    generator.seed(seed);
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
    //    if(gotspike)
    //cout<<"e_traclk: "<<e_track<<endl;

    return release*gotspike;
  }

  void Synapse::RecordErr(float err) {
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

  void Synapse::Update(float eta,float regularization) {
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
    

    
  template <int d> float Neuron<d>::a(float *x) {
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

#ifdef CUDA
  __global__ void d_ProcessLayer(Neuron<DIM> *layer,float *x,float
				 delta_t,float process_time,int *spikes) {
    int a = 0;
    int i = blockIdx.x*blockDim.x+threadIdx.x;
 
    for(float t = 0;t<process_time;t+=delta_t) {
      a += layer[i].Process(x,delta_t);
    }

    spikes[i] = a;


  }


  __global__ void d_SumSpikes(int *spikes,int *answer) {
    //WARNING: It is CALLER's responsibility to make sure that spikes is
    //padded with zeros if necessary
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


#endif

  void RecordErr(Neuron<DIM> *layer, int size,float err) {
    for(int i=0;i<size;i++) {
      layer[i].RecordErr(err);
    }
  }

#ifdef CUDA
  __global__ void d_RecordErr(Neuron<DIM> *layer,float err) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    layer[i].RecordErr(err);
  }
#endif
  
  void Update(Neuron<DIM> *layer, int size, float eta, float
	      regularization) {
    for(int i=0;i<size;i++) {
      layer[i].Update(eta,regularization);
    }
  }

#ifdef CUDA
  
  __global__ void d_Update(Neuron<DIM> *layer, float eta, float
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
    


  __global__ void d_AverageValue(Neuron<DIM> *layer, float *x,float *averages) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    averages[i] = layer[i].average(x);
  }

#endif

  float AverageValue(Neuron<DIM> *layer,int size, float *x) {
    float a = 0;
    for(int i=0;i<size;i++) {
      a += layer[i].average(x);
    }
    return a;
  }



#ifdef CUDA


  template <int d, int size> void GPU_Manager<d,size>::SendToDevice(Neuron<d> *N,int a_size) {
    //for now we'll require that size be a power of 2.
    size = a_size;
    while(a_size % 2==0)
      a_size /= 2;
    if(a_size != 1) {
      cout<<"MUST BE A POWER OF TWO!\nEXITING"<<endl;
      exit(1);
    }
    
    cudaMalloc(&d_N,size*sizeof(Neuron<d>));
    cudaMalloc(&d_spikes,size*sizeof(int));
    cudaMalloc(&d_addspikes,size/2*sizeof(int));
    cudaMalloc(&d_averages,size*sizeof(float));
    cudaMalloc(&d_addaverages,size/2*sizeof(float));
    cudaMalloc(&d_x,d*sizeof(float));

    cudaMemcpy(d_N,N,size*sizeof(Neuron<d>),cudaMemcpyHostToDevice);
  }
  
  template <int d,int size> int GPU_Manager<d,size>::AddSpikes(void) {

    int threadsperblock = size>=256?256:size;
    int blocks = size>=256?size/256:1;


    int *d_answer;
    int h_answer;
    cudaMalloc(&d_answer,sizeof(int));


    if(blocks != 1)
      d_SumSpikes<<<blocks,threadsperblock>>>(d_spikes,d_addspikes);
    else
      d_SumSpikes<<<blocks,threadsperblock>>>(d_spikes,d_answer);


    int itemsleft = blocks;
    int offset = 0;


    while(itemsleft>1) {
      itemsleft = blocks;
      threadsperblock = itemsleft>=256?256:itemsleft;
      blocks = itemsleft>=256?itemsleft/256:1;
      if(itemsleft != 1)
	d_SumSpikes<<<blocks,threadsperblock>>>(d_addspikes+offset,d_addspikes+offset+itemsleft);
      else
	d_SumSpikes<<<blocks,threadsperblock>>>(d_addspikes+offset,d_answer);
      offset += itemsleft;
    }

    cudaMemcpy(&h_answer,d_answer,sizeof(int),cudaMemcpyDeviceToHost);

    return h_answer;
  }

  //I'm scared of template kernels, so we'll do it this way first
  template <int d,int size> float GPU_Manager<d,size>::AddAverages(void) {

    int threadsperblock = size>=256?256:size;
    int blocks = size>=256?size/256:1;


    float *d_answer;
    float h_answer;
    cudaMalloc(&d_answer,sizeof(float));


    if(blocks != 1)
      d_SumAverages<<<blocks,threadsperblock>>>(d_averages,d_addaverages);
    else
      d_SumAverages<<<blocks,threadsperblock>>>(d_averages,d_answer);


    int itemsleft = blocks;
    int offset = 0;


    while(itemsleft>1) {
      itemsleft = blocks;
      threadsperblock = itemsleft>=256?256:itemsleft;
      blocks = itemsleft>=256?itemsleft/256:1;
      if(itemsleft != 1)
	d_SumAverages<<<blocks,threadsperblock>>>(d_addaverages+offset,d_addaverages+offset+itemsleft);
      else
	d_SumAverages<<<blocks,threadsperblock>>>(d_addaverages+offset,d_answer);
      offset += itemsleft;
    }

    cudaMemcpy(&h_answer,d_answer,sizeof(float),cudaMemcpyDeviceToHost);

    return h_answer;
  }

  template <int d,int size> float GPU_Manager<d,size>::ProcessLayer(float *x,float delta_t,float process_time) {

    cudaMemcpy(d_x,x,d*sizeof(float),cudaMemcpyHostToDevice);
    
    int threadsperblock = size>256?256:size;
    int blocks = size>256?size/256:1;

    d_ProcessLayer<<<blocks,threadsperblock>>>(d_N,d_x,delta_t,process_time,d_spikes);

    int spikes = AddSpikes();

    return spikes/process_time;
  }

  template <int d,int size> float GPU_Manager<d,size>::AverageValue(float *x) {    
    cudaMemcpy(d_x,d,d*sizeof(float),cudaMemcpyHostToDevice);

    int threadsperblock = size>256?256:size;
    int blocks = size>256?size/256:1;

    d_AverageValue<<<blocks,threadsperblock>>>(d_N,d_x,d_averages);

    return AddAverages();
  }

  //these are basically wrappers around kernel calls
  template <int d,int size> void GPU_Manager<d,size>::RecordErr(float err) {
    nn
      int threadsperblock = size>256?256:size;
    int blocks = size>256?size/256:1;

    d_RecordErr<<<blocks,threadsperblock>>>(d_N,err);
  }

  template <int d,int size> void GPU_Manager<d,size>::Update(float eta,float regularization) {
    
    int threadsperblock = size>256?256:size;
    int blocks = size>256?size/256:1;

    d_Update<<<blocks,threadsperblock>>>(d_N,eta,regularization);
  }    

	     
	     
#endif



};	     
#endif
