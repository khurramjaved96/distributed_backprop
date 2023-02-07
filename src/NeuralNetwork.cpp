//
// Created by Khurram Javed on 2023-02-07.
//

#include "../include/NeuralNetwork.h"
#include <exception>
#include <iostream>

NeuralNetwork::NeuralNetwork(int input_neurons, int neurons, int synapses, int seed) : mt(seed) {
  std::uniform_real_distribution<float> weight_sampler(-0.1, 0.1);
  std::uniform_int_distribution<int> connections(0, neurons);
  for (int i = 0; i < input_neurons; i++) {
    auto n = new LinearNeuron();
    n->SetIsInput(true);
    this->list_of_neurons.push_back(n);
  }
  for (int i = 0; i < neurons - input_neurons; i++) {
    auto n = new ReluNeuron();
    this->list_of_neurons.push_back(n);
  }
  for (int i = 0; i < synapses; i++) {
    auto s = new Synapse(list_of_neurons[connections(mt)], list_of_neurons[connections(mt)], weight_sampler(mt), 1e-3);
    list_of_synapses.push_back(s);
  }
}

void NeuralNetwork::SetInputNeurons(std::vector<float> observation) {
  auto it = this->list_of_neurons.begin();
  int index = 0;
  while ((*it)->IsInput() && it != this->list_of_neurons.end()) {
    if (index >= observation.size()) {
      std::cout << "Passing fewer observations than input neurons\nb";
      exit(1);
    }
    (*it)->SetValue(observation[index]);
    index++;
  }
}

float NeuralNetwork::Forward() {
  for (auto &list_of_neuron: this->list_of_neurons) {
    list_of_neuron->Forward();
  }
}

void NeuralNetwork::Backward() {
  for (auto &list_of_neuron: this->list_of_neurons) {
    list_of_neuron->Backward();
  }
}
