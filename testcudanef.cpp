
#include"cuda_nef.h"
#include<iostream>
using namespace std;
using NEF::Synapse;
using NEF::Neuron;
int main(int argc,char*argv[]) {
  NEF::Random r;

  int c = 0;

  for(int i=0;i<10000;i++) {
    c += r.flipcoin(0.3333);
  }

  cout<<"measured prob: "<<((float)c)/10000.0<<endl;

  return 0;
}
