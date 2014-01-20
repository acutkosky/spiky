
#include"cuda_nef.h"
#include<cmath>
using namespace std;

real Error(real p, real target) {
  return (p-target)*(p-target);
}


real Target(real x) {
  return x;
}



void Setup(int layersize, real weightval, real eta,NEF_layer &layer) {
  layer.layer.clear();
  NEF_neuron* pneuron;
  Synapse* pexcite;
  Synapse* pinhibit;
  for(int i=0;i<layersize;i++) {
    pexcite = new Synapse(weight,1, 0.0,
    pneuron = new NEF_neuron(
