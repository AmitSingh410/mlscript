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
        .def("__getitem__", [](const Tensor &t, py::tuple index_tuple) -> py::object {
            if (index_tuple.size() != 2) {
                throw py::index_error("Slicing requires two dimensions (e.g., [rows, cols]).");
            }

            Slice row_slice, col_slice;

            // Handle the first dimension (rows)
            if (py::isinstance<py::slice>(index_tuple[0])) {
                py::slice r_slice = index_tuple[0].cast<py::slice>();
                size_t start = 0, stop = 0, step = 0, slicelength = 0;
                r_slice.compute(static_cast<size_t>(t.mat.rows()), &start, &stop, &step, &slicelength);
                row_slice = Slice{static_cast<py::ssize_t>(start), static_cast<py::ssize_t>(stop), static_cast<py::ssize_t>(step)};
            } else {
                py::ssize_t r = index_tuple[0].cast<py::ssize_t>();
                row_slice = Slice{r, r + 1, 1};
            }

            // Handle the second dimension (columns)
            if (py::isinstance<py::slice>(index_tuple[1])) {
                py::slice c_slice = index_tuple[1].cast<py::slice>();
                size_t start = 0, stop = 0, step = 0, slicelength = 0;
                c_slice.compute(static_cast<size_t>(t.mat.cols()), &start, &stop, &step, &slicelength);
                col_slice = Slice{static_cast<py::ssize_t>(start), static_cast<py::ssize_t>(stop), static_cast<py::ssize_t>(step)};
            } else {
                py::ssize_t c = index_tuple[1].cast<py::ssize_t>();
                col_slice = Slice{c, c + 1, 1};
            }

            // Check if it's single element access
            if (row_slice.step == 1 && (row_slice.stop == row_slice.start + 1) &&
                col_slice.step == 1 && (col_slice.stop == col_slice.start + 1)) {
                return py::cast(t.get_element(row_slice.start, col_slice.start));
            }

            return py::cast(t.slice(row_slice, col_slice));
        })
        .def("__getitem__", [](const Tensor &t, py::object index) -> py::object {
            if (py::isinstance<py::slice>(index)) {
                py::slice r_slice = index.cast<py::slice>();
                size_t start = 0, stop = 0, step = 0, slicelength = 0;
                r_slice.compute(static_cast<size_t>(t.mat.rows()), &start, &stop, &step, &slicelength);
                Slice row_slice = {static_cast<py::ssize_t>(start), static_cast<py::ssize_t>(stop), static_cast<py::ssize_t>(step)};
                Slice col_slice = {0, t.mat.cols(), 1}; // Full column slice
                return py::cast(t.slice(row_slice, col_slice));
            }
            return py::cast(t.get_row(index.cast<long>()));
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