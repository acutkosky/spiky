#ifndef CUDA_NEF_H
#define CUDA_NEF_H

#define CUDA

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

namespace NEF {

  __host__ __device__ float dotp(float *a, float* b,int d);

  __host__ __device__ unsigned TausStep(unsigned &z, int S1, int S2, int S3, unsigned M);

  __host__ __device__ unsigned LCGStep(unsigned &z, unsigned A, unsigned C);

  __host__ __device__ float HybridTaus(unsigned &z1,unsigned &z2, unsigned &z3, unsigned &z4);

  struct Random {
    unsigned z1,z2,z3,z4;
    __host__ __device__ int flipcoin(float bias);
  };



  __host__ __device__ float Pval(float q);


  struct Synapse {
    float grad;
    float lastgrad;
    float lastdelta;
    float q;
    float p;
    float e_track;
    int e_count;
    float pert_track;
    float pertsq_track;
    float corr_track;
    float err_track;
    int count;
    int spike_count;

    Random randomizer;

    
    __host__ __device__ int Process(int gotspike);

    __host__ __device__ void RecordErr(float err);
    
    __host__ __device__ void Update(float eta,float regularization);

    __host__ __device__ void DeterministicDeltaRecordErr(float err,float rate);

    __host__ __device__ void DeltaUpdate(float eta, float regularization,float sign);
    __host__ __device__ void DeltaRecordErr(float err, float time);
  };



  template <int d> struct Neuron {
    float tau_ref;
    float tau_RC;
    float J_th;
    float e[d];
    float egrad[d];
    float alpha;
    float J_bias;
    Random randomizer;
    Synapse Pos;
    Synapse Neg;
    
    __host__ __device__ float a(float *x);
    __host__ __device__ float a_postdot(float x);
    __host__ __device__ float numericalD(float* x,float epsilon);
    __host__ __device__ int Process(float *x,float delta_T);
    __host__ __device__ void RecordErr(float err);
    __host__ __device__ void Update(float eta,float regularization);
    __host__ __device__ float average(float *x);
    __host__ __device__ float average_postdot(float x);
    __host__ __device__ int dimension(void);
    __device__ __host__ void DeltaUpdate(float eta, float regularization);
    __host__ __device__ void DeltaRecordErr(float err, float time);
    __host__ __device__ void DeterministicDeltaRecordErr(float err,float *x);
  };





  template <int DIM> float ProcessLayer(Neuron<DIM> *layer, int size,float *x,float delta_t,float
		     process_time);


  template <int DIM> void RecordErr(Neuron<DIM> *layer, int size,float err);

  template <int DIM> void Update(Neuron<DIM> *layer, int size, float eta, float
	      regularization);

  template <int DIM> float AverageValue(Neuron<DIM> *layer,int size, float *x);
#ifdef CUDA
  template <int d> struct GPU_Manager {
    unsigned int size;
    int *d_spikes;
    int *d_addspikes;

    float *d_averages;
    float *d_addaverages;

    Neuron<d> *d_N;
    float *d_x;

    void SendToDevice(Neuron<d> *N,unsigned int a_size); 

    void SendSpikes(int *tosend);

    int AddSpikes(void);
    float AddAverages(float* d_toadd);

    int GetSize(void);

    float ProcessLayer(float *x,float delta_t,float process_time);

    float AverageValue(float * x);

    void AverageValue_Multi(float * xvals, float* averages, int num);

    void RecordErr(float err);

    void Update(float eta,float regularization);

    void DeltaUpdate(float eta, float regularization);

    void DeterministicDeltaRecordErr(float err,float *x);
    void DeltaRecordErr(float err,float time);

  };
#endif

};
  
#endif
