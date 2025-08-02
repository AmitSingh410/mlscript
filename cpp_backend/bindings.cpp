#include "pybind11/pybind11.h"
#include <pybind11/stl.h>
#include "evaluator.hpp"



namespace py = pybind11;

PYBIND11_MODULE(mlscript, m) {
    py::class_<Evaluator>(m, "Evaluator")
        .def(py::init<>())
        .def("eval_expr", &Evaluator::eval_expr)
        .def("set_var", &Evaluator::set_var)
        .def("get_var", &Evaluator::get_var)
        .def("eval_op", &Evaluator::eval_op);
}