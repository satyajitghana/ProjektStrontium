#include "NeuralNet.hpp"
#include "Neuron.hpp"

#include <vector>
#include <cmath>
#include <iostream>

// Constructor of the Neural NeuralNetwork
NeuralNet::NeuralNet(const std::vector<unsigned> &topology) {
    // Get the Number of Layers
    unsigned num_layers = topology.size();

    // Create the Layers and fill them with neurons
    for (unsigned layer_idx = 0 ; layer_idx < num_layers ; layer_idx++) {
        this -> layers.push_back(Layer());

        // number of outputs from this layer to the next layer is the number of neurons in the
        // next layer, if the layer is the output layer, then the number of outputs are zeroz
        unsigned num_outputs = layer_idx == num_layers - 1 ? 0 : topology.at(layer_idx + 1);

        // Now fill the layer with neurons
        for (unsigned neuron_idx = 0 ; neuron_idx <= topology.at(layer_idx) ; neuron_idx++) {
            Layer& last_layer = this -> layers.back();
            last_layer.push_back(Neuron(num_outputs, neuron_idx));
        }

        // Set the Bias Node Value to be 1.0
        this -> layers.back().back().set_output_val(1.0);
    }
}

void NeuralNet::feed_forward(const std::vector<double>& input_vals) {
    // Attach the input vals to the first layer
    for (unsigned i = 0 ; i < input_vals.size() ; i++) {
        this -> layers.at(0).at(i).set_output_val(input_vals.at(i));
    }

    // Forward Propagate this value
    for (unsigned curr_layer = 1 ; curr_layer < layers.size() ; curr_layer++) {
        Layer& prev_layer = layers.at(curr_layer-1);

        // for(Neuron& neuron : layers.at(curr_layer)) {
        //     neuron.feed_forward(prev_layer);
        // }
        for (unsigned n = 0 ; n < this -> layers.at(curr_layer).size() - 1 ; n++) {
            this -> layers.at(curr_layer).at(n).feed_forward(prev_layer);
        }
    }
}

void NeuralNet::back_prop(const std::vector<double>& target_vals) {
   // Calculate the overall net error (RMS)
   Layer& output_layer = this -> layers.back();

   // RMS Error
   double error = 0.0;

   for (unsigned n = 0 ; n < target_vals.size() ; n++) {
      double delta = target_vals.at(n) - output_layer.at(n).get_output_val();
      error += delta * delta;
   }

   error /= output_layer.size() - 1;
   error = sqrt(error); // RMS
   this -> error = error;

   std::cout << "ERROR : " << error << std::endl;

   // Calculate Output Layer Gradients
   // The size is one less to ignore the bias layer
   for (unsigned n = 0 ; n < output_layer.size() - 1 ; n++) {
       output_layer.at(n).calc_output_grads(target_vals.at(n));
   }

   // Calculate Hidden Layer Gradients
   for (unsigned curr_layer = this -> layers.size() - 2 ; curr_layer > 0 ; curr_layer--) {
       Layer& hidden_layer = this -> layers.at(curr_layer);
       Layer& next_layer = this -> layers.at(curr_layer+1);

       for (unsigned n = 0 ; n < hidden_layer.size() ; n++) {
           hidden_layer.at(n).calc_hidden_grads(next_layer);
       }
   }

   // Update the Weights for all the layers from output to first hidden layer
    for (unsigned curr_layer = this -> layers.size() - 1 ; curr_layer > 0 ; curr_layer--) {
        Layer& layer = this -> layers.at(curr_layer);
        Layer& prev_layer = this -> layers.at(curr_layer-1);

        // for (Neuron& neuron: layer) {
        //     neuron.update_input_weights(prev_layer);
        // }

        for (unsigned n = 0 ; n < layer.size() - 1; n++) {
            layer.at(n).update_input_weights(prev_layer);
        }
    }
}

void NeuralNet::get_results(std::vector<double> &result_vals) const {
    result_vals.clear();

    for (auto neuron : this -> layers.back()) {
        result_vals.push_back(neuron.get_output_val());
    }
}

std::ostream& NeuralNet::print(std::ostream& out) const {

    out << "Neural Network Details" << std::endl;
    out << "Layers : ";
    out << "[ ";
    for (const Layer& layer: this -> layers) {
        out << layer.size() << ", ";
    }
    out << "] ";

    return out;
}


