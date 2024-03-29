#include <iostream>
#include <vector>
#include <math.h>

#include "NeuralNet.hpp"

std::ostream& operator<<(std::ostream& _mystream, const NeuralNet& mynet) {
    return mynet.print(_mystream);
}

int main(int, char**) {
    // Test the XOR Neural Network
    std::vector<unsigned> topology({2, 4, 1});
    NeuralNet my_neural_net(topology);

    std::cout << my_neural_net << std::endl;

    std::vector<double> input_vals, target_vals, result_vals;

    for (unsigned t_pass = 0 ; t_pass < 2000 ; t_pass++) {

        std::cout << std::endl;
        std::cout << "PASS " << t_pass+1 << std::endl;

        double x = rand() % 2;
        double y = rand() % 2;

        double res = int(x) xor int(y);

        std::cout << "X1 : " << x << " X2 : " << y << std::endl;
        std::cout << "Y1 : " << res << std::endl;

        input_vals.clear();
        input_vals.assign({x, y});
        my_neural_net.feed_forward(input_vals);

        result_vals.clear();
        my_neural_net.get_results(result_vals);
        std::cout << "[ ";
        for (unsigned o = 0 ; o < result_vals.size() - 1 ; o++) {
            std::cout << result_vals.at(o) << " ";
        }
        // for (double val : result_vals) {
        //     std:: cout <<  val << " ";
        // }
        std::cout << "] ";
        std::cout << std::endl;

        target_vals.clear();
        target_vals.assign({res});
        my_neural_net.back_prop(target_vals);
    }
}
