
#include"cuda_nef.cu"
#include"cuda_nef_host.cpp"
#include<iostream>
#include<ctime>
#include<boost/random.hpp>
#include<cmath>
using namespace std;
using namespace NEF;
using namespace boost::random;
//ok let's learn the identity function... maybe hopefully

#define DIM 4
#define SIZE 32






template <int d> float RMSE(GPU_Manager<d> manager,float* xvals,int numvals,float (*targetfunc) (float *))
{
  float e = 0;
  int c = 0;

  for(int i=0;i<numvals;i++) {
    float tval = targetfunc(xvals);
    float dval = manager.AverageValue(xvals);
    xvals += DIM;
    e += (tval-dval)*(tval-dval);
    c++;
  }

  return sqrt(e/c);
}

float target(float *x) {
  //float r=0;
  return x[0];
  /*
  for(int k=0;k<DIM;k++) {
    r+=x[k];
  }
  return r;
  */
  //return *x;
  //return 400*sin(3.141592654 * (*x)/400.0);
}

float error(float d,float *x,float (*targetfunc) (float *)) {
  float t = targetfunc(x);
  return -(t-d)*(t-d);
}

int main(int argc,char*argv[]) {
  boost::random::mt19937 generator;
  generator.seed(time(0));  

  Neuron<DIM> N[SIZE];// = CreateNeuron<1>(1,time(0),236254,34213,524325,52622465);

  GPU_Manager<DIM> manager;

  FillLayer(N,SIZE);

  for(int i=0;i<DIM;i++) {
    for(int j=0;j<SIZE/DIM;j++) {
      for(int k=0;k<DIM;k++) {
	N[9*i+j].e[k] = 0.0;
      }
      N[9*i+j].e[i] = (float)(2*((generator())%2))-1.0;
      cout<<"set to: "<<N[9*i+j].e[i]<<endl;
    }
  }


  manager.SendToDevice(N,SIZE);
  /*
  int test[256];
  for(int i=0;i<256;i++) {
    test[i] = 2;
  }
  
  manager.SendSpikes(test);
  cout<<"sum is: "<<manager.AddSpikes()<<endl;
  return 0;
  */
  /*  
  float xv = 200.0;
  float a = 0;
  for(int i=0;i<100;i++) {
    for(int j=0;j<10;j++) {
      //a = ProcessLayer(N,SIZE,&x,0.0001,0.1);
      //RecordErr(N,SIZE,-(a-x)*(a-x));
      a += manager.ProcessLayer(&xv,0.0001,0.2);
      //manager.RecordErr(-(a-x)*(a-x));
    }
    //Update(N,SIZE,0.000000003,0.01);
    //float average = AverageValue(N,SIZE,&x);
    //manager.Update(0.000000003,0.01);
    float average = manager.AverageValue(&xv);
    cout<<"i: "<<i<<" val: "<<a/10<<" average: "<<average<<endl;
    a = 0.0;
  }

  return 0;

*/  
  /*
  for(float i=-400;i<400;i+=10) {
    cout<<"i: "<<i<<" dim: "<<manager.AverageValue(&i)<<endl;
  }
  
  
  return 0;

  */
  
  

  
  boost::random::uniform_real_distribution<float> distribution_x(-400,400);

  float test[100][DIM];
  for(int i=0;i<100;i++) {
    for(int k=0;k<DIM;k++)
      test[i][k] = distribution_x(generator);
  }


  float delta_t = 0.0001;
  float average_time = 0.2;  
  float x_period = 1.0;
  float update_period = 1.0;


  float recordtime = 60.0;
  float totaltime = 60*60*24;

  float eta = 0.0000001;
  float regularization = 0.01;
  

  float t = 0.0;
  float x[DIM];
  float update_mark = t;
  float average_mark = t;
  float record_mark = t;
  
  cout<<"average time: "<<average_time<<" x_period: "<<x_period<<" update period: "<<update_period<<" recordtime: "<<recordtime<<endl;
  cout<<"numneurons: "<<SIZE<<" eta: "<<eta<<" regularization: "<<regularization<<endl;
  randvalue(x,DIM,400.0,generator());
  /*
  for(int k=0;k<DIM;k++)
    x[k] = distribution_x(generator);
  */
  while(t<totaltime) {

    float a = manager.ProcessLayer(x,delta_t,average_time);
    t+=average_time;
    float eval = error(a,x,target);
    manager.RecordErr(eval);

    if(t>average_mark+x_period) {
      average_mark = t;
      randvalue(x,DIM,400.0,generator());
      /*
      for(int k=0;k<DIM;k++)
	x[k] = distribution_x(generator);
      */
    }    
    if(t>update_mark+update_period) {
      update_mark = t;
      manager.Update(eta,regularization);
      //cout<<"time: "<<t<<" Current Value: "<<x<<" Target: "<<target(&x)<<" Current Decode: "<<manager.AverageValue(&x)<<" RMSE: "<<RMSE<DIM>(manager,test,100,target)<<endl;
    }
    if(t>record_mark+recordtime) {
      record_mark = t;
      cout<<"time: "<<t<<" Current Value: "<<x[0]<<" Target: "<<target(x)<<" Current Decode: "<<manager.AverageValue(x)<<" RMSE:"<<RMSE<DIM>(manager,(float*)test,100,target)<<endl;
    }
   
  }



  return 0;
  
}
