#include "pybind11/pybind11.h"
#include <pybind11/stl.h>
#include <variant> // Required for std::variant automatic conversion
#include "evaluator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mlscript, m) {
    m.doc() = "mlscript C++ core engine";

    py::class_<Evaluator>(m, "Evaluator")
        .def(py::init<>())
        // Bind the new, renamed methods from evaluator.hpp
        .def("assign_variable", &Evaluator::assign_variable)
        .def("get_variable", &Evaluator::get_variable)
        .def("evaluate", &Evaluator::evaluate)
        
        // Expose the new scope management functions
        .def("enter_scope", &Evaluator::enter_scope)
        .def("exit_scope", &Evaluator::exit_scope);
}