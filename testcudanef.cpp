
#include"cuda_nef.cpp"
#include<iostream>
#include<ctime>
#include<random>
using namespace std;
using namespace NEF;
//ok let's learn the identity function... maybe hopefully

#define DIM 1

float RMSE(Neuron<DIM> *layer,int size,float* xvals,int numvals,float (*targetfunc) (float *))
{
  float e = 0;
  int c = 0;

  for(int i=0;i<numvals;i++) {
    float tval = targetfunc(xvals);
    float dval = AverageValue(layer,size,xvals);
    xvals += DIM;
    e += (tval-dval)*(tval-dval);
    c++;
  }

  return sqrt(e/c);
}

float target(float *x) {
  return *x;
}


int main(int argc,char*argv[]) {
  NEF::Random r;

  Neuron<1> N = CreateNeuron<1>(1,time(0),236254,34213,524325,52622465);


  std::mt19937 generator;
  generator.seed(time(0));

  std::uniform_real_distribution<float> distribution_x(-400,400);


  float t = 0;
  float x = 300;
  for(int i=0;i<1000;i++) {
    float a = ProcessLayer(&N,1,&x,0.0001,0.1);
    t+=0.1;
    float eval = -(a-x)*(a-x);
    RecordErr(&N,1,eval);
    if(i%10==0) {
      Update(&N,1,0,0);
      cout<<"time: "<<t<<" Current Value: "<<AverageValue(&N,1,&x)<<endl;
    }
    
  }
  
  return 0;
}
