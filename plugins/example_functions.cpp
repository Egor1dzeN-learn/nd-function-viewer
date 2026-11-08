// Example of custom functions. Copy this file, change the classes and run:
//     python native/build_plugins.py

#include <cmath>

#include "nd_function.hpp"

// Any dimension from 1 to 12, the global minimum is known and marked on the plot.
class ExampleBowl : public nfv::Function {
public:
    std::string name() const override { return "Example: bowl"; }
    int min_dimension() const override { return 1; }
    int max_dimension() const override { return 12; }
    nfv::Bounds bounds(int dimension) const override { return nfv::box(dimension, -5.0, 5.0); }

    double evaluate(const nfv::Point &x) const override {
        double sum = 0.0;
        for (double coordinate: x) sum += coordinate * coordinate;
        return sum;
    }

    bool minimum(int dimension, nfv::Point &point, double &value) const override {
        point.assign(dimension, 0.0);
        value = 0.0;
        return true;
    }
};

NFV_REGISTER_FUNCTION(ExampleBowl)

// Only 2D, each variable has its own range, the minimum is not specified.
class ExampleRipple : public nfv::Function {
public:
    std::string name() const override { return "Example: ripple"; }
    int min_dimension() const override { return 2; }
    int max_dimension() const override { return 2; }
    nfv::Bounds bounds(int) const override { return {{-6.0, 6.0}, {-3.0, 3.0}}; }

    double evaluate(const nfv::Point &x) const override {
        return std::sin(x[0]) * std::cos(x[1]) + 0.1 * x[0];
    }
};

NFV_REGISTER_FUNCTION(ExampleRipple)
