
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

#define DIM 1
#define SIZE 256






template <int d> float RMSE(GPU_Manager<d> manager,float* xvals,int numvals,float (*targetfunc) (float *))
{
  float e = 0;
  int c = 0;

  for(int i=0;i<numvals;i++) {
    float tval = targetfunc(xvals);
    float dval = manager.AverageValue(xvals);
    xvals += DIM;
    //cout<<"tval: "<<tval<<" dval: "<<dval<<endl;
    e += (tval-dval)*(tval-dval);
    c++;
  }

  return sqrt(e/c);
}


template <int d> float Noisy_RMSE(GPU_Manager<d> manager,float* xvals,int numvals,float (*targetfunc) (float *),float delta_t,float time)
{
  float e = 0;
  int c = 0;

  for(int i=0;i<numvals;i++) {
    float tval = targetfunc(xvals);
    float dval = manager.ProcessLayer(xvals,delta_t,time);
    xvals += DIM;
    //cout<<"tval: "<<tval<<" dval: "<<dval<<endl;
    e += (tval-dval)*(tval-dval);
    c++;
  }

  return sqrt(e/c);
}


float target(float *x) {
  //float r=0;
  return 400*sin(3.141592654*(x[0]/400));
  
  /*for(int k=0;k<DIM;k++) {
    r+=x[k];
  }
  return 2*r;//4*400*sin(3.141592654*(r/800));
		   //return 4*sqrt(r);
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

  boost::random::uniform_real_distribution<float> distribution_x(-400,400);


  Neuron<DIM> *N = new Neuron<DIM>[SIZE];
  if(N==0) {
    cout<<"couldn't malloc!\n";
    exit(1);
  }
  //Neuron<DIM> N[SIZE];// = CreateNeuron<1>(1,time(0),236254,34213,524325,52622465);

  GPU_Manager<DIM> manager;

  cout<<"haven't dont anything yet...\n";
  FillLayer(N,SIZE,generator);

  cout<<"filled layer\n";
  /*  
  for(int i=0;i<DIM;i++) {
    for(int j=0;j<SIZE/DIM;j++) {
      for(int k=0;k<DIM;k++) {
	N[9*i+j].e[k] = 0.0;
      }
      N[9*i+j].e[i] = (float)(2*((generator())%2))-1.0;
      cout<<"set to: "<<N[9*i+j].e[i]<<endl;
    }
  }
  */

  manager.SendToDevice(N,SIZE);

  cout<<"sent to device\n";
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
  
  

  


  float test[100][DIM];
  for(int i=0;i<100;i++) {
    randvalue(test[i],DIM,400.0,generator());
    //    for(int k=0;k<DIM;k++)
    //test[i][k] = distribution_x(generator);
  }


  float delta_t = 0.0001;
  float average_time = 0.2;  
  float x_period = 0.2;
  float update_period = 1.0;


  float recordtime = 60.0;
  float totaltime = 60*30;

  float eta_delta = 0.0001;
  float regularization_delta = 0.01;//0.01;//0.1;//0.0001;

  float eta_cor = 0.000001;
  float regularization_cor = 0.01;//0.0001;
  

  float t = 0.0;
  float x[DIM];
  float update_mark = t;
  float average_mark = t;
  float record_mark = t;
  
  cout<<"average time: "<<average_time<<" x_period: "<<x_period<<" update period: "<<update_period<<" recordtime: "<<recordtime<<endl;
  cout<<"numneurons: "<<SIZE<<" eta: "<<eta_delta<<" regularization: "<<regularization_delta<<endl;
  randvalue(x,DIM,400.0,generator());
  /*
  for(int k=0;k<DIM;k++)
    x[k] = distribution_x(generator);
  */
  int deltaupdates = 1;
  while(t<totaltime) {

    float a = manager.AverageValue(x);
    //float a = manager.ProcessLayer(x,delta_t,average_time);
    t+=average_time;
    float eval = error(a,x,target);
    if(eval > 10*10) {
      cout<<"shouldn't be here\n";
      //deltaupdates = 0;
      //manager.RecordErr(eval);
    }
    else {
      //deltaupdates = 1;
      //manager.DeltaRecordErr(a-target(x),average_time);
      manager.DeterministicDeltaRecordErr(a-target(x),x);//average_time);//,eta,regularization);
    }
    
    if(t>average_mark+x_period) {
      average_mark = t;
      randvalue(x,DIM,400.0,generator());
    }    
    if(t>update_mark+update_period) {
      update_mark = t;
      if(deltaupdates == 1) {
	float eta = eta_delta;
	if(eval>-60*60)
	  eta = eta_delta;//*0.1;
	manager.DeltaUpdate(eta,regularization_delta);
      }
      else {
	//manager.Update(eta_cor,regularization_cor);
      }
      //cout<<"time: "<<t<<" Current Value: "<<x<<" Target: "<<target(&x)<<" Current Decode: "<<manager.AverageValue(&x)<<" RMSE: "<<RMSE<DIM>(manager,test,100,target)<<endl;
    }
    if(t>record_mark+recordtime) {
      record_mark = t;
      cout<<"time: "<<t<<" Current Value: "<<x[0]<<" Target: "<<target(x)<<" Current Decode: "<<manager.AverageValue(x)<<" True RMSE: "<<RMSE<DIM>(manager,(float*)test,100,target)<<" current mode: "<<deltaupdates;
      cout<<endl;
      //cout<<" Noisy_RMSE:"<<Noisy_RMSE<DIM>(manager,(float*)test,100,target,delta_t,average_time)<<endl;
      //manager.DeltaRecordErr(a-target(x),average_time);
      //manager.DeltaUpdate(0.0,0.0);
    }
   
  }



  return 0;
  
}
