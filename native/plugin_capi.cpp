// C interface to the functions registered in plugins/*.cpp, called from Python via ctypes.
// tf_destroy, tf_dimension, tf_bounds, tf_optimum and tf_evaluate match testfuncs_capi.cpp.

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "nd_function.hpp"

#define TF_API extern "C" __declspec(dllexport)

namespace {

    std::vector<nfv::Factory> &factories() {
        static std::vector<nfv::Factory> list;
        return list;
    }

    // Created on first use, not while the DLL loads, so an exception in a constructor is reported as an error.
    const std::vector<std::unique_ptr<nfv::Function>> &functions() {
        static const std::vector<std::unique_ptr<nfv::Function>> list = [] {
            std::vector<std::unique_ptr<nfv::Function>> created;
            for (const auto &factory: factories()) created.push_back(factory());
            return created;
        }();
        return list;
    }

    const nfv::Function &find_function(const std::string &name) {
        for (const auto &function: functions())
            if (function->name() == name) return *function;
        throw std::invalid_argument("unknown function: " + name);
    }

    struct Problem {
        const nfv::Function *function;
        int dimension;
        nfv::Bounds bounds;
        nfv::Point minimum_point;
        double minimum_value = 0.0;
        bool has_minimum = false;
    };

    Problem make_problem(const nfv::Function &function, int dimension) {
        const std::string name = function.name();
        if (dimension < function.min_dimension() || dimension > function.max_dimension())
            throw std::invalid_argument(name + ": dimension " + std::to_string(dimension) + " is not supported");

        Problem problem{&function, dimension, function.bounds(dimension)};
        if (static_cast<int>(problem.bounds.size()) != dimension)
            throw std::invalid_argument(name + ": bounds() must return one range per variable");
        for (const auto &[lower, upper]: problem.bounds)
            if (!(lower < upper)) throw std::invalid_argument(name + ": a lower bound must be less than the upper bound");

        problem.has_minimum = function.minimum(dimension, problem.minimum_point, problem.minimum_value);
        if (problem.has_minimum) {
            if (static_cast<int>(problem.minimum_point.size()) != dimension)
                throw std::invalid_argument(name + ": minimum() must fill one coordinate per variable");
            for (int i = 0; i < dimension; i++)
                if (problem.minimum_point[i] < problem.bounds[i].first || problem.minimum_point[i] > problem.bounds[i].second)
                    throw std::invalid_argument(name + ": the minimum lies outside the bounds");
        }
        return problem;
    }

    std::string g_last_error;

    template<typename F>
    auto guarded(F &&body, decltype(body()) fallback) -> decltype(body()) {
        try {
            return body();
        } catch (const std::exception &e) {
            g_last_error = e.what();
        } catch (...) {
            g_last_error = "unknown error";
        }
        return fallback;
    }
}

int nfv::register_function(Factory factory) {
    factories().push_back(std::move(factory));
    return 0;
}

TF_API const char *tf_last_error() { return g_last_error.c_str(); }

// Function names separated by '\n'. Returns the required buffer size, or 0 on error.
TF_API int tf_plugin_names(char *buffer, int size) {
    return guarded([&]() -> int {
        std::string joined;
        for (const auto &function: functions()) joined += function->name() + "\n";
        if (buffer && size > static_cast<int>(joined.size())) std::copy(joined.c_str(), joined.c_str() + joined.size() + 1, buffer);
        return static_cast<int>(joined.size()) + 1;
    }, 0);
}

TF_API int tf_plugin_dimensions(const char *name, int *minimum, int *maximum) {
    return guarded([&]() -> int {
        const nfv::Function &function = find_function(name);
        *minimum = function.min_dimension();
        *maximum = function.max_dimension();
        return 1;
    }, 0);
}

TF_API void *tf_create_plugin(const char *name, int dimension) {
    return guarded([&]() -> void * {
        return new Problem(make_problem(find_function(name), dimension));
    }, nullptr);
}

TF_API void tf_destroy(void *handle) { delete static_cast<Problem *>(handle); }

TF_API int tf_dimension(void *handle) { return static_cast<Problem *>(handle)->dimension; }

TF_API void tf_bounds(void *handle, double *lower, double *upper) {
    const auto &bounds = static_cast<Problem *>(handle)->bounds;
    for (size_t i = 0; i < bounds.size(); i++) {
        lower[i] = bounds[i].first;
        upper[i] = bounds[i].second;
    }
}

// Returns 1 and fills point/value if the global minimum is known.
TF_API int tf_optimum(void *handle, double *point, double *value) {
    const auto *problem = static_cast<Problem *>(handle);
    if (!problem->has_minimum) return 0;
    std::copy(problem->minimum_point.begin(), problem->minimum_point.end(), point);
    *value = problem->minimum_value;
    return 1;
}

// points holds n points of dimension consecutive coordinates; results are written to out[n].
TF_API int tf_evaluate(void *handle, const double *points, int n, double *out) {
    return guarded([&]() -> int {
        const auto *problem = static_cast<Problem *>(handle);
        const int dimension = problem->dimension;
        nfv::Point x(dimension);
        for (int i = 0; i < n; i++) {
            std::copy(points + static_cast<size_t>(i) * dimension, points + static_cast<size_t>(i + 1) * dimension, x.begin());
            out[i] = problem->function->evaluate(x);
        }
        return 1;
    }, 0);
}
