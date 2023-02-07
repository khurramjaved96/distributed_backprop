//
// Created by Khurram Javed on 2023-02-07.
//

#ifndef INCLUDE_NEURALNETWORK_H_
#define INCLUDE_NEURALNETWORK_H_

#include <random>
#include <vector>
#include <random>

class Synapse;

class Neuron {
 protected:
  float temp_value;
  bool is_input;
  bool is_output;
  float activation_trace;
  float value;

 public:
  void SetValue(float value);

 protected:
  std::vector<Synapse *> incoming_synapses;
  std::vector<Synapse *> outgoing_synapses;

 public:
  bool IsInput() const;
  void SetIsInput(bool is_input);
  bool IsOutput() const;
  void SetIsOutput(bool is_output);
  [[nodiscard]] float GetValue() const;
  void Sync();
  void UpdateValue();
  virtual float Forward() = 0;
  virtual float Backward() = 0;
  void UpdateTraces();
  void UpdateWeight();
  void AddIncomingSynapse(Synapse *);
  void AddOutgoingSynapse(Synapse *);
  Neuron();
};

class LinearNeuron : public Neuron {
 public:
  float Forward();
  float Backward();
};

class ReluNeuron : public Neuron {
 public:
  float Forward();
  float Backward();
};

class Synapse {
  float weight;
  float gradient_trace;
  Neuron *incoming_neuron;
  Neuron *outgoing_neuron;
  float step_size;

 public:
  Synapse(Neuron* input, Neuron* output, float weight, float step_size);
  float GetWeight() const;
  float GetInputValue();
  void UpdateWeight();
};

class NeuralNetwork {

  std::vector<Neuron *> list_of_neurons;
  std::mt19937 mt;
  std::vector<Synapse *> list_of_synapses;

 public:
  NeuralNetwork(int input_neurons, int width, int synapses, int seed);
  void SetInputNeurons(std::vector<float> observation);
  float Forward();
  void Backward();
  void UpdateWeights();
};



#endif//INCLUDE_NEURALNETWORK_H_
