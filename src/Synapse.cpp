//
// Created by Khurram Javed on 2023-02-07.
//

#include "../include/NeuralNetwork.h"



Synapse::Synapse(Neuron *input, Neuron *output, float weight, float step_size) {
  this->incoming_neuron = input;
  this->outgoing_neuron = output;
  this->weight = weight;
  this->step_size = step_size;
  this->gradient_trace = 0;
  this->incoming_neuron->AddOutgoingSynapse(this);
  this->outgoing_neuron->AddIncomingSynapse(this);
}



float Synapse::GetWeight() const {
  return weight;
}

float Synapse::GetInputValue() {
  return this->incoming_neuron->GetValue();
}