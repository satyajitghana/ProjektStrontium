#include "Neuron.hpp"

#include <cmath>

Neuron::Neuron() {

}

Neuron::Neuron(double num_outputs, unsigned neuron_idx) {
    for(unsigned c = 0 ; c < num_outputs ; c++) {
        this -> output_weights.push_back(Connection());
        this -> output_weights.back().weight = random_weight();
    }

    this -> neuron_idx = neuron_idx;
}

void Neuron::feed_forward(const Layer& prev_layer) {
    double sum = 0.0;

    // perform w_i * x_i
    for (const Neuron& prev_neuron: prev_layer) {
        sum += prev_neuron.get_output_val() * prev_neuron.get_connections().at(this -> neuron_idx).weight;
    }

    // Pass it through the activation function
    this -> output_val = Neuron::activation_function(sum);
}

double Neuron::activation_function(double x) {
    return tanh(x);
}

double Neuron::activation_function_derivative(double x) {
    return 1.0 - x * x;
}

void Neuron::calc_output_grads(double target_val) {
    double delta = target_val - this -> output_val;
    this -> gradient = delta * Neuron::activation_function_derivative(this -> output_val);
}

void Neuron::calc_hidden_grads(const Layer& next_layer) {
    double dow = sum_delta_weights(next_layer);
    this -> gradient = dow * Neuron::activation_function_derivative(this -> output_val);
}

double Neuron::sum_delta_weights(const Layer& next_layer) {
    double sum = 0.0;

    for (unsigned n = 0 ; n < next_layer.size() - 1 ; n++) {
        sum += this -> output_weights.at(n).weight * next_layer.at(n).gradient;
    }

    return sum;
}

double Neuron::eta = 0.15; // Net Learning Rate
double Neuron::alpha = 0.5; // Momentum

void Neuron::update_input_weights(Layer& prev_layer) {
    for (unsigned n = 0 ; n < prev_layer.size() ; n++) {
        Neuron& neuron = prev_layer.at(n);

        double old_delta_weight = neuron.get_connections().at(this -> neuron_idx).delta_weight;

        double new_delta_weight = 
            eta * neuron.get_output_val()
                * this -> gradient
                + alpha * old_delta_weight;
        
        neuron.get_connections().at(this -> neuron_idx).delta_weight = new_delta_weight;
        neuron.get_connections().at(this -> neuron_idx).weight += new_delta_weight;
    }
}
