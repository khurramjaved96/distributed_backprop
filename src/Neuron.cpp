//
// Created by Khurram Javed on 2023-02-07.
//

#include "../include/NeuralNetwork.h"

void Neuron::AddIncomingSynapse(Synapse *s) {
  this->incoming_synapses.push_back(s);
}

void Neuron::AddOutgoingSynapse(Synapse *s) {
  this->outgoing_synapses.push_back(s);
}

void Neuron::UpdateValue() {
  temp_value = 0;
  for (auto &in : this->incoming_synapses) {
    temp_value += in->GetInputValue() * in->GetWeight();
  }
}

Neuron::Neuron() {
  this->gradient_for_propogation = 0;
  this->local_gradient_trace = 0;
  this->value = 0;
  this->temp_value = 0;
  this->activation_trace = 0;
  this->is_input = false;
  this->is_output = false;
}

void Neuron::Sync() { this->value = this->temp_value; }
bool Neuron::IsInput() const { return is_input; }
void Neuron::SetIsInput(bool is_input) { Neuron::is_input = is_input; }
bool Neuron::IsOutput() const { return is_output; }

void Neuron::UpdateTraces() {
  this->activation_trace = this->activation_trace * 0.9 + 0.1 * this->value;
  this->local_gradient_trace =
      this->local_gradient_trace * 0.9 + 0.1 * this->Backward();
}

void Neuron::SetIsOutput(bool is_output) { Neuron::is_output = is_output; }
float Neuron::GetValue() const { return value; }
void Neuron::SetValue(float value) { Neuron::value = value; }
float Neuron::GetLocalGradientTrace() const { return local_gradient_trace; }

float LinearNeuron::Forward() { return this->value; }

float LinearNeuron::Backward() { return 1; }

float ReluNeuron::Forward() {
  if (this->value > 0)
    return this->value;
  return 0;
}

float ReluNeuron::Backward() {
  if (this->value > 0) {
    return 1;
  }
  return 0;
}
