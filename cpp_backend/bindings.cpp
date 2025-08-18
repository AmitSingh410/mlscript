#include "pybind11/pybind11.h"
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>
#include <sstream>
#include <variant> 
#include <pybind11/numpy.h>
#include "evaluator.hpp"


namespace py = pybind11;

PYBIND11_MODULE(mlscript, m) {
    m.doc() = "mlscript C++ core engine";
    

    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor", py::buffer_protocol())
        .def(py::init(&std::make_shared<Tensor, const std::vector<std::vector<double>>&>))
        .def(py::init([](py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            if (arr.ndim() != 2) {
                throw std::runtime_error("NumPy array must be 2-dimensional to create a Tensor.");
            }
            auto buf = arr.request();
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> map(
                static_cast<double*>(buf.ptr), buf.shape[0], buf.shape[1]
            );
            return std::make_shared<Tensor>(map);
        }))
        .def_buffer([](Tensor &t) -> py::buffer_info {
            return py::buffer_info(
                t.mat.data(),                               
                sizeof(double),                            
                py::format_descriptor<double>::format(),    
                2,                                          
                { t.mat.rows(), t.mat.cols() },             
                { sizeof(double) * t.mat.cols(),
                  sizeof(double) }
            );
        })
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self * double())
        .def(double() * py::self)
        .def("matmul", &Tensor::matmul)
        .def("sum", &Tensor::sum)
        .def("backward", &Tensor::backward)
        .def_property_readonly("grad", [](const Tensor &t) {
            return py::array_t<double>(
                {t.grad.rows(), t.grad.cols()},
                {sizeof(double) * t.grad.cols(), sizeof(double)},
                t.grad.data()
            );
        })
        .def("__getitem__", [](const Tensor &t, py::tuple index_tuple) -> py::object {
            if (index_tuple.size() > 2) {
                throw py::index_error("Tensor slicing supports at most 2 dimensions.");
            }

            Slice row_slice, col_slice;

            if (py::isinstance<py::slice>(index_tuple[0])) {
                py::slice r_slice = index_tuple[0].cast<py::slice>();
                size_t start = 0, stop = 0, step = 0, slicelength = 0;
                r_slice.compute(static_cast<size_t>(t.mat.rows()), &start, &stop, &step, &slicelength);
                row_slice = Slice{static_cast<py::ssize_t>(start), static_cast<py::ssize_t>(stop), static_cast<py::ssize_t>(step)};
            } else {
                py::ssize_t r = index_tuple[0].cast<py::ssize_t>();
                row_slice = Slice{r, r + 1, 1};
            }

            if (index_tuple.size() == 2) {
                if (py::isinstance<py::slice>(index_tuple[1])) {
                    py::slice c_slice = index_tuple[1].cast<py::slice>();
                    size_t start = 0, stop = 0, step = 0, slicelength = 0;
                    c_slice.compute(static_cast<size_t>(t.mat.cols()), &start, &stop, &step, &slicelength);
                    col_slice = Slice{static_cast<py::ssize_t>(start), static_cast<py::ssize_t>(stop), static_cast<py::ssize_t>(step)};
                } else {
                    py::ssize_t c = index_tuple[1].cast<py::ssize_t>();
                    col_slice = Slice{c, c + 1, 1};
                }
            } else {
                 col_slice = Slice{0, t.mat.cols(), 1}; 
            }


            if (index_tuple.size() == 2 &&
                !py::isinstance<py::slice>(index_tuple[0]) &&
                !py::isinstance<py::slice>(index_tuple[1])) {
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
                Slice col_slice = {0, t.mat.cols(), 1}; 
                return py::cast(t.slice(row_slice, col_slice));
            }
            long r = index.cast<long>();
            if (r < 0) r += t.mat.rows();
            return py::cast(t.get_row(r));
        })
        .def("__repr__", [](const Tensor &t) {
            std::stringstream ss;
            ss << "Tensor(\n" << t.mat;
            if (t.grad.size() > 0 && t.grad.cwiseAbs().sum() > 1e-9) {
                ss << ",\ngrad=\n" << t.grad;
            }
            ss << "\n)";
            return ss.str();
        });


    py::class_<Evaluator>(m, "Evaluator")
        .def(py::init<>())
        .def("assign_variable", &Evaluator::assign_variable)
        .def("get_variable", &Evaluator::get_variable)
        .def("set_grad_enabled", &Evaluator::set_grad_enabled)
        .def("evaluate", &Evaluator::evaluate)
        .def("matmul", &Evaluator::matmul)
        .def("enter_scope", &Evaluator::enter_scope)
        .def("exit_scope", &Evaluator::exit_scope);
}