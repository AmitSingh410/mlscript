#include "pybind11/pybind11.h"
#include <pybind11/stl.h>
#include <variant> // Required for std::variant
#include "evaluator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mlscript, m) {
    py::class_<Evaluator>(m, "Evaluator")
        .def(py::init<>())
        // eval_expr still takes a string and returns a Value (variant)
        .def("eval_expr", &Evaluator::eval_expr)
        // set_var now takes a Value (variant) from Python
        .def("set_var", &Evaluator::set_var)
        .def("get_var", &Evaluator::get_var)
        // eval_op now takes two Values (variants)
        .def("eval_op", &Evaluator::eval_op);
}