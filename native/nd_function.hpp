// Base class for custom C++ functions.
//
// Put a .cpp file into plugins/, derive a class from nfv::Function, register it with
// NFV_REGISTER_FUNCTION and run: python native/build_plugins.py
// A complete example is in plugins/example_functions.cpp.

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nfv {

    using Point = std::vector<double>;
    using Bounds = std::vector<std::pair<double, double>>;  // {lower, upper} of each variable

    class Function {
    public:
        virtual ~Function() = default;

        virtual std::string name() const = 0;               // shown in the list, must be unique
        virtual int min_dimension() const = 0;              // the viewer supports dimensions 1 to 12
        virtual int max_dimension() const = 0;
        virtual Bounds bounds(int dimension) const = 0;     // above 4D all variables must share one range
        virtual double evaluate(const Point &x) const = 0;  // x.size() is the dimension

        // Override to mark the global minimum on the plot: fill point and value and return true.
        virtual bool minimum(int dimension, Point &point, double &value) const { return false; }
    };

    // The same range for all variables.
    inline Bounds box(int dimension, double lower, double upper) { return Bounds(dimension, {lower, upper}); }

    using Factory = std::function<std::unique_ptr<Function>()>;
    int register_function(Factory factory);
}

#define NFV_CONCAT_(a, b) a##b
#define NFV_CONCAT(a, b) NFV_CONCAT_(a, b)

// Makes a class derived from nfv::Function (with a default constructor) appear in the Plugins tab.
#define NFV_REGISTER_FUNCTION(Class)                                       \
    [[maybe_unused]] static const int NFV_CONCAT(nfv_registered_, __COUNTER__) = \
        nfv::register_function([] { return std::unique_ptr<nfv::Function>(new Class()); });
