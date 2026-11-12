#pragma once

struct Neuron {
	std::vector<double> weights;
	double bias;
	double output = 0.0;
	double delta = 0.0; // This is the blame score for back propagation
};

struct Layer {
	std::vector<Neuron> neurons;
	std::string name;
};

struct Network {
	std::vector<Layer> layers;
};

double randomWeight();

void create(Layer& layer, int neurons, int weights);

Network setup(const std::vector<int>& topology);

void printStats(const Network& network);

double sigmoid(double z);

double total_error(const std::vector<double>& outputs, const std::vector<double>& targets);

void backpropagate(Network& network, const std::vector<double>& targets, double learning_rate);

std::vector<double> forward_pass(Network& network, std::vector<double> inputs);

void save_network(const std::vector<int>& instructions, const Network& n);

int main(void);
