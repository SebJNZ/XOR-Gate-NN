#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include "main.h"

void constructNetwork(std::ifstream& networkData, Network& n) {
	for (std::size_t i = 1; i < n.layers.size(); i++) {
		Layer& start = n.layers.at(i);
		for (std::size_t j = 0; j < start.neurons.size(); j++) {
			Neuron& n = start.neurons.at(j);
			std::string line;
			std::getline(networkData, line);

			std::istringstream string_stream(line);
			std::string temp;
			
			n.weights.clear();

			if (string_stream >> temp) {
				n.bias = std::stod(temp);

				while(string_stream >> temp) {
					n.weights.push_back(std::stod(temp));
				}
			} else {
				continue;
			}
		}
	}
}

int load_file(Network& n) {
	// Load File
	std::ifstream networkData("network.nn");
	
	if (!networkData.is_open()) {
		std::cerr << "Error Opening File\n";
		return 1;
	}
	
	// Get first line -> Our Speifications
	std::string specifics;
	std::getline(networkData, specifics);
	
	// Create input string stream, and temp variable for stream of variables
    std::istringstream string_stream(specifics);
    std::string temp;
	
	// Vector of instructions we extract 
	std::vector<int> instructions; 

	while (string_stream >> temp) { // Loop throughout the input stream
		instructions.push_back(std::stod(temp)); // Convert string to integer, and push to vector.
	}
	
    n = setup(instructions); // Define our network structure.

	constructNetwork(networkData, n);

	return 0;
}

int main(void) {
	Network n;
	if (load_file(n) != 0) return 1;

	std::cout << "Loaded\n\n";

	printStats(n);

	double in1, in2;

	while (true) {
		std::cout << "Enter two binary numbers: ";
		std::cin >> in1 >> in2;

		if ((in1 != 0 && in1 != 1) || (in2 != 0 && in2 != 1)) {
			continue;
		}

		std::vector<double> input;
		input.push_back(in1);
		input.push_back(in2); 

		std::vector<double> outcome = forward_pass(n, input);

		for (double d : outcome) {
			std::cout << d << "\n";
		}

		std::cout << "\n\n";
	}

	return 0;
}
