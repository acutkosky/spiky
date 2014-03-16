#ifndef CUDA_NEF_CPP
#define CUDA_NEF_CPP

#include"cuda_nef.cu.h"
#include<cmath>
#include<stdio.h>

#include<iostream>
using namespace std;

#ifdef CUDA
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}
#endif

#ifndef CUDA
#define __device__
#define __host__
#define __global__
#endif



namespace NEF {
  __device__ __host__ float dotp(float *a,float *b,int d) {
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


  __device__ __host__ float Pval(float q) {
    return 1.0/(1.0+exp(-q));
  }
    

  __host__ __device__ int Synapse::Process(int gotspike) {
    //no conditionals - faster to not branch on a gpu (I think)

    int release = randomizer.flipcoin(p);
    
    spike_count+= gotspike;

    e_track += gotspike*(release - p);
    e_count += gotspike;
    //    if(gotspike)
    //cout<<"e_traclk: "<<e_track<<endl;

    return release*gotspike;
  }



  __host__ __device__ void Synapse::DeltaUpdate(float eta, float regularization,float sign) {
    //float rate = spike_count / time;
    float grad_est = grad*p*(1-p);//-regularization*p;
    //printf("grad: %f spike_count: %d, err: %f r %f  p: %f\n",eta*grad,spike_count,err,rate,p);
    float curgrad = grad_est*sign+regularization*p*(1-p);
    lastdelta = -eta*(curgrad);// *   lastdelta/(curgrad - lastgrad);

    lastgrad = curgrad;//-grad_est*sign-regularization*p*(1-p);
    
    q += lastdelta;
    //printf("q: %f  gradest: %f\n",eta*(-grad_est*sign-regularization*p),grad);
    if(q>10)
      q =10;
    if(q<-10)
      q = -10;

    grad = 0.0;
    //spike_count = 0;
    p = Pval(q);
    //printf("p: %f\n",p);
  }

  __host__ __device__ void Synapse::DeterministicDeltaRecordErr(float err,float rate) {
    //printf("adding: %f err: %f rate: %f\n",err*rate,err,rate);
    grad += err*rate;
    spike_count = 0;
  }
  
  __host__ __device__ void Synapse::DeltaRecordErr(float err, float time) {
    float rate = spike_count / time;
    grad += err*rate;
    spike_count = 0;
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
    //printf("count: %d\n",count);
    e_track = 0.0;
    e_count = 0;

  }

  __device__ __host__  void Synapse::Update(float eta,float regularization) {
    float avpertsq = pertsq_track/count;
    float avpert = pert_track/count;
    float avcorr = corr_track/count;
    float averr = err_track/count;
    p = Pval(q);
    //printf("count: %d\n",count);
    //printf("original p: %f\n",p);
    if(avpertsq != avpert*avpert) {

      float estimate = (avcorr - averr*avpert)/(avpertsq-avpert*avpert);
      //printf("estimate: %f\n",estimate);
      q += eta*(estimate*p*(1-p) - regularization *p);
      if(q<-10)
	q = -10;
      if(q>10)
	q = 10;
    }
    pert_track = 0;
    pertsq_track = 0;
    corr_track = 0;
    err_track = 0;
    count = 0;
    e_track = 0;
    e_count = 0;

    p = Pval(q);
    //printf("new p: %f\n",p);
  }
    

  template <int d> __device__ __host__ float Neuron<d>::numericalD(float* x,float epsilon) {
    float p,n;
    float xval = dotp(e,x,dimension());
    if(alpha*(xval+epsilon)+J_bias <= J_th)
      p=0;
    else
      p = 1.0/(tau_ref-tau_RC*log(1.0-J_th/(alpha*(xval+epsilon)+J_bias)));

    if(alpha*(xval-epsilon)+J_bias <= J_th)
      n=0;
    else
      n = 1.0/(tau_ref-tau_RC*log(1.0-J_th/(alpha*(xval-epsilon)+J_bias)));
    return (p-n)/(2*epsilon);

  }
  
  template <int d> __device__ __host__ float Neuron<d>::a_postdot(float x) {
    //printf("got: %f\n",x);
    if(alpha*x+J_bias <= J_th) {
      //printf("0.0\n");
      return 0.0;
    }
    //printf("%f\n", 1.0/(tau_ref-tau_RC*log(1.0-J_th/(alpha*x+J_bias))));
    return 1.0/(tau_ref-tau_RC*log(1.0-J_th/(alpha*x+J_bias)));
  }
  
  template <int d> __device__ __host__ float Neuron<d>::a(float *x) {
    //printf("working with: %f\n",dotp(e,x,dimension()));
    if(alpha*dotp(e,x,dimension())+J_bias <= J_th) {
      //printf("0.0\n");
      return 0.0;
    }
    //printf("%f\n", 1.0/(tau_ref-tau_RC*log(1.0-J_th/(alpha*dotp(e,x,dimension())+J_bias))));
    return 1.0/(tau_ref-tau_RC*log(1.0-J_th/(alpha*dotp(e,x,dimension())+J_bias)));
  }
  
  template <int d> __device__ __host__ float Neuron<d>::average(float *x) {
    Pos.p = Pval(Pos.q);
    Neg.p = Pval(Neg.q);
    return a(x)*(Pos.p-Neg.p);
  }

  
  template <int d> __device__ __host__ float Neuron<d>::average_postdot(float x) {
    Pos.p = Pval(Pos.q);
    Neg.p = Pval(Neg.q);
    return a_postdot(x)*(Pos.p-Neg.p);
  }

  template <int d> __device__ __host__ int Neuron<d>::dimension(void) {
    return d;
  }
  

  template <int d> __device__ __host__ int Neuron<d>::Process(float *x,float delta_T) {
    float rate = a(x);
    int spike = randomizer.flipcoin(rate*delta_T);
    return Pos.Process(spike)-Neg.Process(spike);
  }

  template <int d> __device__ __host__ void Neuron<d>::RecordErr(float err) {
    Pos.RecordErr(err);
    Neg.RecordErr(err);
  }

  
  template <int d> __device__ __host__ void Neuron<d>::DeltaRecordErr(float err,float time) {
    Pos.DeltaRecordErr(err,time);
    Neg.DeltaRecordErr(err,time);
  }

  
  template <int d> __device__ __host__ void Neuron<d>::DeterministicDeltaRecordErr(float err,float *x) {

    float postdot = dotp(e,x,dimension());

    float aval = a_postdot(postdot);

    //float D = (a_postdot(postdot+0.00001)-aval)/(0.00001);

    Pos.DeterministicDeltaRecordErr(err,aval);
    Neg.DeterministicDeltaRecordErr(err,aval);
    /*
    float w = Pos.p-Neg.p;

    for(int i=0;i<dimension();i++) {
      egrad[i] += -err*w*D*x[i];
      }
    */

  }

  template <int d> __device__ __host__ void Neuron<d>::Update(float eta,float regularization) {
    Pos.Update(eta,regularization);
    Neg.Update(eta,regularization);
  }

  template <int d> __device__ __host__ void Neuron<d>::DeltaUpdate(float eta, float regularization) {
    Pos.DeltaUpdate(eta,regularization,1);
    Neg.DeltaUpdate(eta,regularization,-1);

    /*
    for(int i=0;i<dimension();i++) {
      e[i] += eta*(egrad[i] - regularization*e[i]);
      egrad[i] = 0.0;
    }
    */
    
  }

  
  template <int DIM> float ProcessLayer(Neuron<DIM> *layer, int size,float *x,float delta_t,float
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
  template <int DIM> __global__ void d_ProcessLayer(Neuron<DIM> *layer,float *x,float
				  delta_t,float process_time,int *spikes) {
    int a = 0;
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
 
    for(float t = 0.0;t<process_time;t+=delta_t) {
      a += layer[i].Process(x,delta_t);
    }

    spikes[i] = a;


  }

  __global__ void d_SumSpikes(int *spikes,int *answer) {
    //WARNING: It is CALLER's responsibility to make sure that spikes is
    //padded with zeros if necessary
    unsigned int offset = blockIdx.x*2*blockDim.x+threadIdx.x;
    unsigned int id = threadIdx.x;

    extern __shared__ int spikes_sums[];
    spikes_sums[id] = spikes[offset]+spikes[offset+blockDim.x];

    //printf("sum: %d, id: %d, blocksize: %d\n",spikes_sums[id],id,blockIdx.x);

    __syncthreads();

    for(unsigned int i=blockDim.x/2;i>0;i>>= 1) {
      if(id<i) {
	spikes_sums[id] += spikes_sums[id+i];
      }
      __syncthreads();
    }

    /*
    if(id<32) {
      spikes_sums[id] += spikes_sums[id+32];
      __syncthreads();
      spikes_sums[id] += spikes_sums[id+16];
      __syncthreads();
      spikes_sums[id] += spikes_sums[id+8];
      __syncthreads();
      spikes_sums[id] += spikes_sums[id+4];
      __syncthreads();
      spikes_sums[id] += spikes_sums[id+2];
      __syncthreads();
      spikes_sums[id] += spikes_sums[id+1];
    }
    */
    if(id == 0) {
      answer[blockIdx.x] = spikes_sums[0];
      //printf("answer: %d stored at index: %d\n",spikes_sums[0],blockIdx.x);
    }
  }


#endif

  template <int DIM> void RecordErr(Neuron<DIM> *layer, int size,float err) {
    for(int i=0;i<size;i++) {
      layer[i].RecordErr(err);
    }
  }

#ifdef CUDA

  template <int DIM> __global__ void d_RecordErr(Neuron<DIM> *layer,float err) {
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    layer[i].RecordErr(err);
  }

  template <int DIM> __global__ void d_DeltaRecordErr(Neuron<DIM> *layer,float err,float time) {
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    layer[i].DeltaRecordErr(err,time);
  }

  template <int DIM> __global__ void d_DeterministicDeltaRecordErr(Neuron<DIM> *layer,float err,float *x) {
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    layer[i].DeterministicDeltaRecordErr(err,x);
  }

#endif
  
  template <int DIM> void Update(Neuron<DIM> *layer, int size, float eta, float
	      regularization) {
    for(int i=0;i<size;i++) {
      layer[i].Update(eta,regularization);
    }
  }

#ifdef CUDA
  
  template <int DIM> __global__ void d_Update(Neuron<DIM> *layer, float eta, float
	      regularization) {
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;    
    layer[i].Update(eta,regularization);
  }


  template <int DIM> __global__ void d_DeltaUpdate(Neuron<DIM> *layer,float eta, float regularization) {
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    layer[i].DeltaUpdate(eta,regularization);
  }


  __global__ void d_SumAverages(float *averages,float *answer) {

    unsigned int offset = blockIdx.x*2*blockDim.x+threadIdx.x;
    unsigned int id = threadIdx.x;

    extern __shared__ float average_sums[];
    average_sums[id] = averages[offset]+averages[offset+blockDim.x];

    __syncthreads();

    for(unsigned int i=blockDim.x/2;i>0;i>>= 1) {
      if(id<i) {
	average_sums[id] += average_sums[id+i];
      }
      __syncthreads();
    }
    /*

    if(id<32) {
      average_sums[id] += average_sums[id+32];
      average_sums[id] += average_sums[id+16];
      average_sums[id] += average_sums[id+8];
      average_sums[id] += average_sums[id+4];
      average_sums[id] += average_sums[id+2];
      average_sums[id] += average_sums[id+1];
    }
    */
    if(id == 0) {
      answer[blockIdx.x] = average_sums[0];
    }
  }


  __global__ void d_SumFloatVecs(float* tosum,int stride, int length,int reducelength,float *answers) {
    unsigned int row = blockIdx.x;//blockIdx.y*blockDim.y;
    unsigned int column = threadIdx.x;//blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int id = threadIdx.x;

    int imax = length/reducelength;
    __shared__ float sums[64];
    sums[column]=0.0;
    for(int i=1;i<imax;i++) {
      sums[column] += tosum[row*stride+column+reducelength*i];
    }


    __syncthreads();

    for(unsigned int i=blockDim.x/2;i>0;i>>= 1) {
      if(id<i) {
	sums[id] += sums[id+i];
      }
      __syncthreads();
    }

    if(id == 0)
      answers[row] = sums[0];
  }



  template <int DIM> __global__ void d_AverageValue(Neuron<DIM> *layer, float *x,float *averages) {
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    averages[i] = layer[i].average(x);
  }


  template <int DIM> __global__ void d_AverageValues_Multi(Neuron<DIM> *layer, float *x, float *averages,int num) {
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y+threadIdx.y;
    //printf("i: %d j: %d\n",i,j);
    averages[i+j*num] = layer[i].average_postdot(x[i+j*num]);
  }


#endif

  template <int DIM> float AverageValue(Neuron<DIM> *layer,int size, float *x) {
    float a = 0;
    for(int i=0;i<size;i++) {
      a += layer[i].average(x);
    }
    return a;
  }



#ifdef CUDA


  template <int d> void GPU_Manager<d>::SendToDevice(Neuron<d> *N,unsigned int a_size) {
    //for now we'll require that size be a power of 2.
    size = a_size;
    while(a_size % 2==0)
      a_size /= 2;
    if(a_size != 1) {
      cout<<"MUST BE A POWER OF TWO!\nEXITING"<<endl;
      exit(1);
    }
    
    gpuErrchk(cudaMalloc(&d_N,size*sizeof(Neuron<d>)));
    gpuErrchk(cudaMalloc(&d_spikes,size*sizeof(int)));
    gpuErrchk(cudaMalloc(&d_addspikes,size/2*sizeof(int)));
    gpuErrchk(cudaMalloc(&d_averages,size*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_addaverages,size/2*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_x,d*sizeof(float)));

    gpuErrchk(cudaMemcpy(d_N,N,size*sizeof(Neuron<d>),cudaMemcpyHostToDevice));
    
    

  }
  
  template <int d> int GPU_Manager<d>::AddSpikes(void) {
    unsigned int groupsize = 256;
    unsigned int threadsperblock = size>=2*groupsize?groupsize:size/2;
    unsigned int blocks = size>=2*groupsize?size/(2*groupsize):1;

    int *d_answer;
    int h_answer;
    gpuErrchk(cudaMalloc(&d_answer,sizeof(int)));


    if(blocks != 1)
      d_SumSpikes<<<blocks,threadsperblock,threadsperblock*sizeof(int)>>>(d_spikes,d_addspikes);
    else
      d_SumSpikes<<<blocks,threadsperblock,threadsperblock*sizeof(int)>>>(d_spikes,d_answer);


    unsigned int itemsleft = blocks;
    unsigned int offset = 0;


    while(itemsleft>1) {

      threadsperblock = itemsleft>=2*groupsize?groupsize:itemsleft/2;
      blocks = itemsleft>=2*groupsize?itemsleft/(2*groupsize):1;
      if(blocks > 1)
	d_SumSpikes<<<blocks,threadsperblock,threadsperblock*sizeof(int)>>>(d_addspikes+offset,d_addspikes+offset+itemsleft);
      else
	d_SumSpikes<<<blocks,threadsperblock,threadsperblock*sizeof(int)>>>(d_addspikes+offset,d_answer);
      offset += itemsleft;
      itemsleft = blocks;
    }

    gpuErrchk(cudaMemcpy(&h_answer,d_answer,sizeof(int),cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_answer));
    return h_answer;
  }



  //I'm scared of template kernels, so we'll do it this way first
  template <int d> float GPU_Manager<d>::AddAverages(float* d_toadd) {
    unsigned int groupsize = 512;
    unsigned int threadsperblock = size>=2*groupsize?groupsize:size/2;
    unsigned int blocks = size>=2*groupsize?size/(2*groupsize):1;
    /*
      unsigned int threadsperblocky = num>=16?16:num;
      unsigned int blocksy = num>=16?num/16:1;
    */

    float *d_answer;
    float h_answer;


    //gpuErrchk(cudaMalloc(&d_addaverages,sizeof(float)*size/2 * num));
    gpuErrchk(cudaMalloc(&d_answer,sizeof(float)));

    /*
      dim3 dimBLOCK(threadsperblockx,threadsperblocky);
      dim3 dimGRID(blocksx,blocksy);
    */

    if(blocks != 1)
      d_SumAverages<<<blocks,threadsperblock,threadsperblock*sizeof(float)>>>(d_toadd,d_addaverages);
    else
      d_SumAverages<<<blocks,threadsperblock,threadsperblock*sizeof(float)>>>(d_toadd,d_answer);


    unsigned int itemsleft = blocks;
    unsigned int offset = 0;


    while(itemsleft>1) {

      threadsperblock = itemsleft>=2*groupsize?groupsize:itemsleft/2;
      blocks = itemsleft>=128?2*groupsize/(2*groupsize):1;
      if(blocks != 1)
	d_SumAverages<<<blocks,threadsperblock,threadsperblock*sizeof(float)>>>(d_addaverages+offset,d_addaverages+offset+itemsleft);
      else
	d_SumAverages<<<blocks,threadsperblock,threadsperblock*sizeof(float)>>>(d_addaverages+offset,d_answer);
      offset += itemsleft;
      itemsleft = blocks;
    }

    gpuErrchk(cudaMemcpy(&h_answer,d_answer,sizeof(float),cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_answer));
    return h_answer;
  }

  template <int d> int GPU_Manager<d>::GetSize(void) {
    return d;
  }

  template <int d> float GPU_Manager<d>::ProcessLayer(float *x,float delta_t,float process_time) {

    gpuErrchk(cudaMemcpy(d_x,x,d*sizeof(float),cudaMemcpyHostToDevice));
    
    unsigned int threadsperblock = size>256?256:size;
    unsigned int blocks = size>256?size/256:1;

    d_ProcessLayer<<<blocks,threadsperblock>>>(d_N,d_x,delta_t,process_time,d_spikes);

    int spikes = AddSpikes();

    return spikes/process_time;
  }

  template <int d> float GPU_Manager<d>::AverageValue(float *x) {    
    gpuErrchk(cudaMemcpy(d_x,x,d*sizeof(float),cudaMemcpyHostToDevice));

    unsigned int threadsperblock = size>256?256:size;
    unsigned int blocks = size>256?size/256:1;

    d_AverageValue<<<blocks,threadsperblock>>>(d_N,d_x,d_averages);

    return AddAverages(d_averages);
  }

  //ok we want to parallelize batchmode...

  //NUM MUST BE DIVISIBLE BY 256 RIGHT NOW
  template <int d> void GPU_Manager<d>::AverageValue_Multi(float *xvals, float* averages,int num) {

    if(num>256 && num%256 !=0) {
      cout<<"NO! Number of examples must be divisible by 256!\n";
      exit(1);
    }
    //cout<<"averaging\n";
    float *d_xvals;
    float *d_averages_multi;
    gpuErrchk(cudaMalloc(&d_xvals,size*num*sizeof(float)));    
    gpuErrchk(cudaMalloc(&d_averages_multi,size*num*sizeof(float)));
    gpuErrchk(cudaMemcpy(d_xvals,xvals,size*num*sizeof(float),cudaMemcpyHostToDevice));

    unsigned int threadsperblockx = size>256?256:size;
    unsigned int blocksx = size>256?size/256:1;

    unsigned int threadsperblocky = num>256?256:num;
    unsigned int blocksy = num>256?num/256:1;

    dim3 dimBlock(threadsperblockx,threadsperblocky);
    dim3 dimGrid(blocksx,blocksy);

    d_AverageValues_Multi<<<dimGrid,dimBlock>>>(d_N,d_xvals,d_averages_multi,size);


    float * d_averages;
    gpuErrchk(cudaMalloc(&d_averages,num*sizeof(float)));


    unsigned int reducelength = size>64?64:1;
    dim3 dimSumBlock(reducelength,1);
    dim3 dimSumGrid(1,num);
    //cout<<"summing...\n";
    d_SumFloatVecs<<<num,reducelength>>>(d_averages_multi,size,size,reducelength,d_averages);
    //cout<<"done\n";

    gpuErrchk(cudaMemcpy(averages,d_averages,num*sizeof(float),cudaMemcpyDeviceToHost));
    
    gpuErrchk(cudaFree(d_averages_multi));
    gpuErrchk(cudaFree(d_xvals));

    /*
    for(int i=0;i<num;i++) {
      averages[i] = AddAverages(d_averages_multi+i*size);
      }
    */

  }






  //these are basically wrappers around kernel calls
  template <int d> void GPU_Manager<d>::RecordErr(float err) {

    unsigned int threadsperblock = size>256?256:size;
    unsigned int blocks = size>256?size/256:1;

    d_RecordErr<<<blocks,threadsperblock>>>(d_N,err);
  }

  template <int d> void GPU_Manager<d>::DeltaRecordErr(float err,float time) {

    unsigned int threadsperblock = size>256?256:size;
    unsigned int blocks = size>256?size/256:1;

    d_DeltaRecordErr<<<blocks,threadsperblock>>>(d_N,err,time);
  }

  template <int d> void GPU_Manager<d>::DeterministicDeltaRecordErr(float err,float *x) {


    gpuErrchk(cudaMemcpy(d_x,x,d*sizeof(float),cudaMemcpyHostToDevice));
    unsigned int threadsperblock = size>256?256:size;
    unsigned int blocks = size>256?size/256:1;

    d_DeterministicDeltaRecordErr<<<blocks,threadsperblock>>>(d_N,err,d_x);
  }


  template <int d> void GPU_Manager<d>::Update(float eta,float regularization) {
    
    unsigned int threadsperblock = size>256?256:size;
    unsigned int blocks = size>256?size/256:1;

    d_Update<<<blocks,threadsperblock>>>(d_N,eta,regularization);
  }    

  template <int d> void GPU_Manager <d>::DeltaUpdate(float eta, float regularization) {
    unsigned int threadsperblock = size>256?256:size;
    unsigned int blocks = size>256?size/256:1;
    
    d_DeltaUpdate<<<blocks,threadsperblock>>>(d_N,eta,regularization);
  }

  template <int d> void GPU_Manager<d>::SendSpikes(int *tosend) {
    gpuErrchk(cudaMemcpy(d_spikes,tosend,size*sizeof(int),cudaMemcpyHostToDevice));
  }

	     
#endif

};	     

#endif
