#pragma once

#include "Neuron.hpp"

#include <vector>
#include <iostream>

class NeuralNet {
    public:
        NeuralNet(const std::vector<unsigned>& topology);
        void feed_forward(const std::vector<double>& inputVals);
        void back_prop(const std::vector<double>& targetVals);
        void get_results(std::vector<double> &resultVals) const;

        virtual std::ostream& print(std::ostream &_mystream) const;
    private:
        std::vector<Layer> layers;
        double error;
};