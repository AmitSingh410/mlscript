#include "pybind11/pybind11.h"
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>
#include <sstream>

#include <variant> // Required for std::variant automatic conversion
#include "evaluator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mlscript, m) {
    m.doc() = "mlscript C++ core engine";

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<std::vector<double>>&>())
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self * double())
        .def(double() * py::self)
        .def("matmul", &Tensor::matmul)
        .def("__repr__", [](const Tensor &t) {
            std::stringstream ss;
            ss << t.mat;
            return "Tensor(\n" + ss.str() + "\n)";
        });


    py::class_<Evaluator>(m, "Evaluator")
        .def(py::init<>())
        .def("assign_variable", &Evaluator::assign_variable)
        .def("get_variable", &Evaluator::get_variable)

        // IMPORTANT: Specific overloads MUST be defined before the generic one.
        .def("evaluate", py::overload_cast<const std::string&, const Tensor&, const Tensor&>(&Evaluator::evaluate), "Evaluates two Tensors")
        .def("evaluate", py::overload_cast<const std::string&, const Tensor&, double>(&Evaluator::evaluate), "Evaluates Tensor and scalar")
        .def("evaluate", py::overload_cast<const std::string&, double, const Tensor&>(&Evaluator::evaluate), "Evaluates scalar and Tensor")
        .def("evaluate", py::overload_cast<const std::string&, const Value&, const Value&>(&Evaluator::evaluate), "Evaluates non-tensor types")

        .def("matmul", py::overload_cast<const Tensor&, const Tensor&>(&Evaluator::matmul))
        
        .def("enter_scope", &Evaluator::enter_scope)
        .def("exit_scope", &Evaluator::exit_scope);
}