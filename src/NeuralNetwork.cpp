//
// Created by Khurram Javed on 2023-02-07.
//

#include "../include/NeuralNetwork.h"
#include <exception>
#include <iostream>

NeuralNetwork::NeuralNetwork(int input_neurons, int neurons, int synapses,
                             int seed)
    : mt(seed) {
  std::uniform_real_distribution<float> weight_sampler(-0.1, 0.1);
  std::uniform_int_distribution<int> connections(0, neurons-1);
  for (int i = 0; i < input_neurons; i++) {
    auto n = new LinearNeuron();
    n->SetIsInput(true);
    this->list_of_neurons.push_back(n);
  }
  for (int i = 0; i < neurons - input_neurons; i++) {
    auto n = new ReluNeuron();
    this->list_of_neurons.push_back(n);
  }
  std::vector<std::vector<int>> matrix;
  for(int i = 0; i < neurons; i++){
    std::vector<int> row;
    for(int j = 0; j < neurons; j++){
      row.push_back(0);
    }
    matrix.push_back(row);
  }
  for (int i = 0; i < synapses; i++) {
    int i_index = connections(mt);
    int j_index = connections(mt);
    if(matrix[i_index][j_index] != 1 and j_index > input_neurons) {
      std::cout << i_index << " -> " << j_index << std::endl;
      matrix[i_index][j_index] = 1;
      auto s = new Synapse(list_of_neurons[i_index],
                           list_of_neurons[j_index], weight_sampler(mt),
                           1e-3);
      list_of_synapses.push_back(s);
    }
  }
}

void Neuron::PropagateGradient() {
  float new_grad_to_prop = 0;
  for (auto &s : this->outgoing_synapses) {
    if (s->GetOutgoingNeuron()->IsOutput()) {
      new_grad_to_prop += 1;
    } else {
      new_grad_to_prop += s->GetWeight() *
                          s->GetOutgoingNeuron()->gradient_for_propogation *
                          s->GetOutgoingNeuron()->GetLocalGradientTrace();
    }
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
  for (auto &list_of_neuron : this->list_of_neurons) {
    list_of_neuron->Forward();
  }
  return 0;
}

void NeuralNetwork::Backward() {
  for (auto &list_of_neuron : this->list_of_neurons) {
    list_of_neuron->Backward();
  }
}
