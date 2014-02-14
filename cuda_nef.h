

namespace NEF {

  float dotp(float *a, float* b,int d);

  unsigned TausStep(unsigned &z, int S1, int S2, int S3, unsigned M);

  unsigned LCGStep(unsigned &z, unsigned A, unsigned C);

  float HybridTaus(unsigned &z1,unsigned &z2, unsigned &z3, unsigned &z4);

  struct Random {
    unsigned z1,z2,z3,z4;
    int flipcoin(float bias);
  };

 

  float Pval(float q);


  struct Synapse {
    float q;
    float p;
    float e_track;
    int e_count;
    float pert_track;
    float pertsq_track;
    float corr_track;
    float err_track;
    int count;

    Random randomizer;


    int Process(int gotspike);

    void RecordErr(float err);
    
    void Update(float eta,float regularization);
  };



  template <int d> struct Neuron {
    float tau_ref;
    float tau_RC;
    float J_th;
    float e[d];
    float alpha;
    float J_bias;
    float weight;
    Random randomizer;
    Synapse Pos;
    Synapse Neg;

    float a(float *x);
    int Process(float *x,float delta_T);
    void RecordErr(float err);
    void Update(float eta,float regularization);
    float average(float *x);
    int dimension(void);

  };



  template <int d> Neuron<d> CreateNeuron(void);

};

  
