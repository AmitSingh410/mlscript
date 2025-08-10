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
        .def("__getitem__", [](const Tensor &t, py::tuple index) {
            if (index.size() != 2) {
                throw py::index_error("Only 2D indexing is supported.");
            }
            long row = index[0].cast<long>();
            long col = index[1].cast<long>();
            return t.get_element(row, col);
        })
        .def("__getitem__", [](const Tensor &t, long row) {
            return t.get_row(row);
        })
        .def("__repr__", [](const Tensor &t) {
            std::stringstream ss;
            ss << t.mat;
            return "Tensor(\n" + ss.str() + "\n)";
        });


    py::class_<Evaluator>(m, "Evaluator")
        .def(py::init<>())
        .def("assign_variable", &Evaluator::assign_variable)
        .def("get_variable", &Evaluator::get_variable)
        .def("evaluate", &Evaluator::evaluate)
        .def("matmul", &Evaluator::matmul)
        .def("enter_scope", &Evaluator::enter_scope)
        .def("exit_scope", &Evaluator::exit_scope);
}