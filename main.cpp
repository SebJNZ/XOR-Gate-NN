#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <fstream>
#include "main.h"

double randomWeight() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

void create(Layer& layer, int neurons, int weights) {
	for (int i = 0; i < neurons; i++) {
		std::vector<double> w;
		
		for (int j = 0; j < weights; j++) {
			w.push_back(randomWeight());
		}
		
		Neuron n = {w, randomWeight(), 0};
		layer.neurons.push_back(n);
	}
}

Network setup(const std::vector<int>& topology) {	
	Network network;
	
	for (size_t layer = 0; layer < topology.size(); layer++) {
		Layer newLayer;
		
		if (layer == 0) {
            newLayer.name = "Input";
            create(newLayer, topology[layer], 1);
        } else if (layer == topology.size() - 1) {
            newLayer.name = "Output";
            create(newLayer, topology[layer], topology[layer - 1]);
        } else {
            newLayer.name = "Hidden " + std::to_string(layer);
            create(newLayer, topology[layer], topology[layer - 1]);
        }
        
        network.layers.push_back(newLayer);
	}
		
	return network;
}

void printStats(const Network& network) {
	for (const Layer& layer : network.layers) {
		std::cout << "Layer: " << layer.name << "\nNeurons: " << layer.neurons.size() << '\n';
		for (const Neuron& neuron : layer.neurons) {
			std::cout << "Neuron Weights: ";
			for (double weight : neuron.weights) {
				std::cout << weight << ' ';
			}
			std::cout << "| Bias: " << neuron.bias << " | Output: " << neuron.output <<"\n\n";
		}
		std::cout << '\n';
	}
}

double sigmoid(double z) {
	return 1.0 / (1.0  + exp(-z));
}

/**
 * Cost Function (Loss Function), this tells the network how wrong those predictions it had are.
 * 
 * This is called the Mean Squared Error Function.
 * 
 * Formila: E = 1/2(y-t)^2
 * 
 * 0.0 means flawless, higher means it is widly off.
 */
double total_error(const std::vector<double>& outputs, const std::vector<double>& targets) {
	double error = 0.0;
	
	for (long unsigned int i = 0; i < outputs.size(); i++) {
		error += 0.5 * std::pow((outputs.at(i) - targets.at(i)), 2);
	}
	
	return error;
}

void backpropagate(Network& network, const std::vector<double>& targets, double learning_rate) {
	Layer& backLayer = network.layers.back();
	Layer& beforeBackLayer = network.layers.at(network.layers.size()-2);
	
	for (std::size_t neuronIndex = 0; neuronIndex < backLayer.neurons.size(); neuronIndex++) {
		Neuron& neuron = backLayer.neurons.at(neuronIndex);
		neuron.delta = (neuron.output - targets.at(neuronIndex)) * neuron.output * (1.0 - neuron.output);
		neuron.bias -= (learning_rate * neuron.delta);
		
		for (std::size_t weightIndex = 0; weightIndex < neuron.weights.size(); weightIndex++) {
			neuron.weights.at(weightIndex) -= (learning_rate * neuron.delta * beforeBackLayer.neurons.at(weightIndex).output);
		}
	}
	
	for (int layerIndex = network.layers.size() - 2; layerIndex > 0; layerIndex--) {
		Layer& currentLayer = network.layers.at(layerIndex);
		
		for (std::size_t neuronIndex = 0; neuronIndex < currentLayer.neurons.size(); neuronIndex++) {
			Neuron& currentNeuron = currentLayer.neurons.at(neuronIndex);
			double blame_sum = 0.0;
			
			// Loop through each of the neurons in the next layer to calculate the blame score.
			Layer& nextLayer = network.layers.at(layerIndex + 1);
			for (std::size_t nextIndex = 0; nextIndex < nextLayer.neurons.size(); nextIndex++) {
				blame_sum += nextLayer.neurons.at(nextIndex).delta * nextLayer.neurons.at(nextIndex).weights.at(neuronIndex);
			}
			
			currentNeuron.delta = blame_sum * currentNeuron.output * (1.0 - currentNeuron.output);
			currentNeuron.bias -= (learning_rate * currentNeuron.delta);
			
			Layer& prevLayer = network.layers.at(layerIndex - 1);
			for (std::size_t weightIndex = 0; weightIndex < currentNeuron.weights.size(); weightIndex++) {
				currentNeuron.weights.at(weightIndex) -= (learning_rate * currentNeuron.delta * prevLayer.neurons.at(weightIndex).output);
			}
		}
	}
}

std::vector<double> forward_pass(Network& network, std::vector<double> inputs) {
	std::vector<Layer>& layers = network.layers;
	
	// Set the values of the front neurons to the inputs
	std::vector<Neuron>& neurons = layers.front().neurons;
	for (std::size_t i = 0; i < neurons.size(); i++) {
		neurons.at(i).output = inputs.at(i);
	}
	
	// Process the hidden neurons, and outputs neurons, multiplying the previous neurons output by the corresponding weight of the current neuron.	
	for (std::size_t i = 1; i < layers.size(); i++) {		
		for (Neuron& neuron : layers.at(i).neurons) {
			double z = 0.0;
			
			std::vector<Neuron>& prevNeurons = layers.at(i-1).neurons;
			for (std::size_t j = 0; j < prevNeurons.size(); j++) {
				z += prevNeurons.at(j).output * neuron.weights.at(j);
			}
			
			z += neuron.bias; // add our bias
			
			neuron.output = sigmoid(z); //  Squash with sigmoid
		}
	}
	
	// Finally return the outputs as a Vector of Doubles.
	std::vector<double> output;
	for (const Neuron& neuron : layers.back().neurons) {
		output.push_back(neuron.output);
	}
	
	return output;
}

void save_network(const std::vector<int>& instructions, const Network& n) {
	std::ofstream outFile("network.nn");
	
	for (std::size_t i = 0; i < instructions.size() - 1; i++) {
		outFile << instructions.at(i) << " "; 
	}
	
	outFile << instructions.back() << std::endl;
	
	for (std::size_t layerIndex = 1; layerIndex < n.layers.size(); layerIndex++) {
		const Layer& layer = n.layers.at(layerIndex);
		
		for (Neuron n : layer.neurons) {
			outFile << n.bias;
			for (double weight : n.weights) {
				outFile << " " << weight;
			}
			outFile << std::endl;
		}
	}
	
	outFile.close();
}

#ifndef NETWORKTESTING
int main(void) {
	srand(time(NULL));
	std::vector<int> instructions = {2, 4, 1};
	Network n = setup(instructions);
	
	//std::vector<double> inputs = {1.0, 0.5};
	std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	
	std::vector<double> expected_output = {0, 1, 1, 0};
	
	std::cout << "Training...\n";
	
	for (uint epoch = 0; epoch < 100000; epoch++) {
		for (int pass = 0; pass < 4; pass++) {
			std::vector<double> current_input = inputs.at(pass);
			std::vector<double> expected;
			expected.push_back(expected_output.at(pass));
			
			std::vector<double> output = forward_pass(n, current_input);
			if (epoch % 200 == 0) {
				std::cout << "Epoch " << epoch << " | Wrongness: " << total_error(output, expected) << " | Output: " << output[0] << "\n";
			}
			backpropagate(n, expected, 0.1);
		}
	}
	
	for (int pass = 0; pass < 4; pass++) {
		std::cout << "\nFinal Output after training for input pair - " << inputs.at(pass).at(0) << " " << inputs.at(pass).at(1) << ": " << forward_pass(n, inputs.at(pass))[0] << "\n";
	}
	
	save_network(instructions, n);
	return 0;
}
#endif