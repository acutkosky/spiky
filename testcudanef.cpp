
#include"cuda_nef.cpp"
#include<iostream>
#include<ctime>
#include<random>
#include<cmath>
using namespace std;
using namespace NEF;
//ok let's learn the identity function... maybe hopefully

#define DIM 1
#define SIZE 50

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
  return 400*sin(3.141592654 * (*x)/400.0);
}

float error(float d,float *x,float (*targetfunc) (float *)) {
  float t = targetfunc(x);
  return -(t-d)*(t-d);
}

int main(int argc,char*argv[]) {
  NEF::Random r;

  Neuron<DIM> N[SIZE];// = CreateNeuron<1>(1,time(0),236254,34213,524325,52622465);

  FillLayer(N,SIZE);

  std::mt19937 generator;
  generator.seed(time(0));
  
  std::uniform_real_distribution<float> distribution_x(-400,400);

  float test[100];
  for(int i=0;i<100;i++) {
    test[i] = distribution_x(generator);
  }


  float delta_t = 0.0002;
  float average_time = 0.1;  
  float x_period = 2.0;
  float update_period = 10.0;


  float recordtime = 60.0;
  float totaltime = 60*60*24;

  float eta = 0.0001;
  float regularization = 0.1;
  

  float t = 0.0;
  float x;
  float update_mark = t;
  float average_mark = t;
  float record_mark = t;
  
  while(t<totaltime) {
    average_mark = t;
    while(t<average_mark+x_period) {

      x = distribution_x(generator);
      float a = ProcessLayer(N,SIZE,&x,delta_t,average_time);
      t+=average_time;
      float eval = error(a,&x,target);
      RecordErr(N,SIZE,eval);
    }
    
    if(t>update_mark+update_period) {
      update_mark = t;
      Update(N,SIZE,eta,regularization);
      //cout<<"time: "<<t<<" Current Value: "<<x<<" Target: "<<target(&x)<<" Current Decode: "<<AverageValue(N,SIZE,&x)<<" RMSE: "<<RMSE(N,SIZE,test,100,target)<<endl;
    }
    if(t>record_mark+recordtime) {
      record_mark = t;
      cout<<"time: "<<t<<" Current Value: "<<x<<" Target: "<<target(&x)<<" Current Decode: "<<AverageValue(N,SIZE,&x)<<" RMSE:"<<RMSE(N,SIZE,test,100,target)<<endl;
    }
    
  }


  
  return 0;
}
