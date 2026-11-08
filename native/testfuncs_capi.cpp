// C interface to the global-opt test functions, called from Python via ctypes.
//
// Function sources:
//   * GCGen (3rd_party/GCGen): Hill, Shekel, Hansen (1D), Grishagin (2D), GKLS (nD) families;
//   * test_functions/classic_functions: classic functions (Rastrigin, Ackley, ...).
//
// A problem is created by one of the tf_create_* functions, which return an opaque handle,
// and released with tf_destroy. On error nullptr / 0 is returned; the message is in tf_last_error().

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "GKLS/GKLSProblem.hpp"
#include "Grishagin/grishagin_function.hpp"
#include "HansenProblemFamily.hpp"
#include "Hill/HillProblem.hpp"
#include "Shekel/ShekelProblem.hpp"
#include "test_functions/classic_functions.h"

#define TF_API extern "C" __declspec(dllexport)

namespace {

    std::string g_last_error;

    struct Problem {
        virtual ~Problem() = default;
        virtual int dimension() const = 0;
        virtual void bounds(std::vector<double> &lo, std::vector<double> &hi) const = 0;
        virtual bool optimum(std::vector<double> &point, double &value) const = 0;
        virtual double evaluate(const std::vector<double> &x) const = 0;
    };

    struct GCGenProblem : Problem {
        std::shared_ptr<IOptProblemFamily> family;  // Hansen problems are owned by the family
        std::unique_ptr<IOptProblem> owned;
        IOptProblem *problem;

        explicit GCGenProblem(IOptProblem *p) : owned(p), problem(p) {}
        GCGenProblem(std::shared_ptr<IOptProblemFamily> f, int index) : family(std::move(f)), problem((*family)[index]) {}

        int dimension() const override { return problem->GetDimension(); }
        void bounds(std::vector<double> &lo, std::vector<double> &hi) const override { problem->GetBounds(lo, hi); }
        bool optimum(std::vector<double> &point, double &value) const override {
            point = problem->GetOptimumPoint();
            value = problem->GetOptimumValue();
            return true;
        }
        double evaluate(const std::vector<double> &x) const override { return problem->ComputeFunction(x); }
    };

    struct ClassicProblem : Problem {
        test_functions::TestFunction function;

        explicit ClassicProblem(test_functions::TestFunction f) : function(std::move(f)) {}

        int dimension() const override { return static_cast<int>(function.dim); }
        void bounds(std::vector<double> &lo, std::vector<double> &hi) const override {
            lo.clear();
            hi.clear();
            for (const auto &[l, h]: function.bounds) {
                lo.push_back(l);
                hi.push_back(h);
            }
        }
        bool optimum(std::vector<double> &point, double &value) const override {
            if (function.x_opt.empty()) return false;
            point = function.x_opt;
            value = function.f_opt;
            return true;
        }
        double evaluate(const std::vector<double> &x) const override { return function(x); }
    };

    template<typename F>
    auto guarded(F &&body, decltype(body()) fallback) -> decltype(body()) {
        try {
            return body();
        } catch (const std::exception &e) {
            g_last_error = e.what();
        } catch (const char *e) {  // GCGen throws string literals
            g_last_error = e;
        } catch (...) {
            g_last_error = "unknown error";
        }
        return fallback;
    }

    void check_index(int index, int first, int last) {
        if (index < first || index > last)
            throw std::out_of_range("problem index must be in [" + std::to_string(first) + ", " + std::to_string(last) + "]");
    }
}

enum Family { HILL = 0, SHEKEL = 1, HANSEN = 2, GRISHAGIN = 3, GKLS = 4 };

TF_API const char *tf_last_error() { return g_last_error.c_str(); }

// family: Family; index: Hill/Shekel 0..999, Hansen 0..19, Grishagin/GKLS 1..100.
// dim, gkls_class (0 = Hard, 1 = Simple), gkls_type (0 = ND, 1 = D, 2 = D2) are used for GKLS only.
TF_API void *tf_create_gcgen(int family, int index, int dim, int gkls_class, int gkls_type) {
    return guarded([&]() -> void * {
        switch (family) {
            case HILL:
                check_index(index, 0, NUM_HILL_PROBLEMS - 1);
                return new GCGenProblem(new THillProblem(index));
            case SHEKEL:
                check_index(index, 0, NUM_SHEKEL_PROBLEMS - 1);
                return new GCGenProblem(new TShekelProblem(index));
            case HANSEN: {
                auto family_ptr = std::make_shared<THansenProblemFamily>();
                check_index(index, 0, family_ptr->GetFamilySize() - 1);
                return new GCGenProblem(family_ptr, index);
            }
            case GRISHAGIN:
                check_index(index, 1, 100);
                return new GCGenProblem(new TGrishaginProblem(index));
            case GKLS:
                check_index(index, 1, 100);
                return new GCGenProblem(new TGKLSProblem(index, dim, static_cast<GKLSClass>(gkls_class),
                                                         static_cast<GKLSFuncionType>(gkls_type)));
            default:
                throw std::invalid_argument("unknown family");
        }
    }, nullptr);
}

TF_API void *tf_create_classic(const char *name, int dim) {
    return guarded([&]() -> void * {
        return new ClassicProblem(test_functions::get_function(name, static_cast<size_t>(dim)));
    }, nullptr);
}

TF_API void tf_destroy(void *handle) { delete static_cast<Problem *>(handle); }

TF_API int tf_dimension(void *handle) { return static_cast<Problem *>(handle)->dimension(); }

TF_API void tf_bounds(void *handle, double *lo, double *hi) {
    std::vector<double> l, h;
    static_cast<Problem *>(handle)->bounds(l, h);
    std::copy(l.begin(), l.end(), lo);
    std::copy(h.begin(), h.end(), hi);
}

// Returns 1 and fills point/value if the global minimum is known.
TF_API int tf_optimum(void *handle, double *point, double *value) {
    return guarded([&]() -> int {
        std::vector<double> p;
        if (!static_cast<Problem *>(handle)->optimum(p, *value)) return 0;
        std::copy(p.begin(), p.end(), point);
        return 1;
    }, 0);
}

// points holds n points of dim consecutive coordinates; results are written to out[n].
TF_API int tf_evaluate(void *handle, const double *points, int n, double *out) {
    return guarded([&]() -> int {
        auto *problem = static_cast<Problem *>(handle);
        const int dim = problem->dimension();
        std::vector<double> x(dim);
        for (int i = 0; i < n; i++) {
            std::copy(points + static_cast<size_t>(i) * dim, points + static_cast<size_t>(i + 1) * dim, x.begin());
            out[i] = problem->evaluate(x);
        }
        return 1;
    }, 0);
}

// Classic function names separated by '\n'. Returns the required buffer size.
TF_API int tf_classic_names(char *buffer, int size) {
    std::string joined;
    for (const auto &name: test_functions::function_names()) joined += name + "\n";
    if (buffer && size > static_cast<int>(joined.size())) std::copy(joined.c_str(), joined.c_str() + joined.size() + 1, buffer);
    return static_cast<int>(joined.size()) + 1;
}
