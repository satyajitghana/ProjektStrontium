#pragma once

#include <vector>
#include <math.h>

class Neuron;

typedef std::vector<Neuron> Layer;

struct Connection {
    double weight;
    double delta_weight;
};

class Neuron {
    public:
        Neuron(unsigned num_outputs, unsigned neuron_idx);

        void feed_forward(const Layer& prev_layer);
        void calc_output_grads(double target_val);
        void calc_hidden_grads(const Layer& next_layer);
        void update_input_weights(Layer& prev_layer);

        // Getters and Setters
        double get_output_val() const { return this -> output_val; };
        void set_output_val(double new_val) { this ->output_val = new_val; };

        std::vector<Connection> get_connections() const { return this -> output_weights; };

        std::vector<Connection>& get_connections_ref() { return this -> output_weights; };

    private:
        static double eta;
        static double alpha;
        static double activation_function(double x);
        static double activation_function_derivative(double x);
        static double random_weight() { return rand() / double(RAND_MAX); };
        double sum_delta_weights(const Layer& next_layer) const;

        double output_val;
        std::vector<Connection> output_weights;
        unsigned neuron_idx;
        double gradient;
};
