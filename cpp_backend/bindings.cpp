#include "pybind11/pybind11.h"
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>
#include <sstream>
#include <variant>
#include <pybind11/numpy.h>
#include "evaluator.hpp"
#include "nn.hpp"

namespace py = pybind11;

// ===================================================================
// Trampoline Classes for Abstract Base Classes
// ===================================================================

class PyModule : public nn::Module {
public:
    using nn::Module::Module; // Inherit constructors
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<Tensor>, nn::Module, forward, input);
    }
};

class PyLoss : public nn::Loss {
public:
    using nn::Loss::Loss; // Inherit constructors
    std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& predictions, const std::shared_ptr<Tensor>& targets) override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<Tensor>, nn::Loss, operator(), predictions, targets);
    }
};

class PyOptimizer : public nn::Optimizer {
public:
    using nn::Optimizer::Optimizer; // Inherit constructors
    void step() override {
        PYBIND11_OVERRIDE_PURE(void, nn::Optimizer, step);
    }
};


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
        .def("__add__", &Tensor::operator+)
        .def("__sub__", &Tensor::operator-)
        .def("__mul__", &Tensor::operator*)
        .def("__truediv__", &Tensor::operator/)

        .def("__mul__", [](const std::shared_ptr<Tensor>& self, double scalar){
            return self->graph_mul_scalar(scalar);
        })
        .def("__rmul__", [](const std::shared_ptr<Tensor>& self, double scalar){
            return operator*(scalar, self);
        })

        .def("matmul", &Tensor::matmul)
        .def("sum", &Tensor::sum)
        .def("softmax", &Tensor::softmax)
        .def("log", &Tensor::log)
        .def("log_softmax", &Tensor::log_softmax)
        .def("backward", &Tensor::backward)
        .def("zero_grad", &Tensor::zero_grad)
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
            // ... (code to parse slice from tuple) ...
            if (py::isinstance<py::slice>(index_tuple[0])) {
                py::slice r_slice = index_tuple[0].cast<py::slice>();
                size_t start, stop, step, slicelength;
                r_slice.compute(t.mat.rows(), &start, &stop, &step, &slicelength);
                row_slice = { (py::ssize_t)start, (py::ssize_t)stop, (py::ssize_t)step };
            } else {
                py::ssize_t r = index_tuple[0].cast<py::ssize_t>();
                row_slice = { r, r + 1, 1 };
            }
            if (index_tuple.size() == 2) {
                if (py::isinstance<py::slice>(index_tuple[1])) {
                    py::slice c_slice = index_tuple[1].cast<py::slice>();
                    size_t start, stop, step, slicelength;
                    c_slice.compute(t.mat.cols(), &start, &stop, &step, &slicelength);
                    col_slice = { (py::ssize_t)start, (py::ssize_t)stop, (py::ssize_t)step };
                } else {
                    py::ssize_t c = index_tuple[1].cast<py::ssize_t>();
                    col_slice = { c, c + 1, 1 };
                }
            } else {
                col_slice = { 0, t.mat.cols(), 1 };
            }
            if (index_tuple.size() == 2 && !py::isinstance<py::slice>(index_tuple[0]) && !py::isinstance<py::slice>(index_tuple[1])) {
                return py::cast(t.get_element(row_slice.start, col_slice.start));
            }
            return py::cast(t.slice(row_slice, col_slice));
        })
        .def("__getitem__", [](const Tensor &t, py::object index) -> py::object {
            if (py::isinstance<py::slice>(index)) {
                py::slice r_slice = index.cast<py::slice>();
                size_t start, stop, step, slicelength;
                r_slice.compute(t.mat.rows(), &start, &stop, &step, &slicelength);
                Slice row_slice = { (py::ssize_t)start, (py::ssize_t)stop, (py::ssize_t)step };
                Slice col_slice = { 0, t.mat.cols(), 1 };
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
        })
        .def_static("random", &Tensor::random, py::arg("rows"), py::arg("cols"));

    py::class_<Evaluator>(m, "Evaluator")
        .def(py::init<>())
        .def("assign_variable", &Evaluator::assign_variable)
        .def("get_variable", &Evaluator::get_variable)
        .def("set_grad_enabled", &Evaluator::set_grad_enabled)
        .def("evaluate", &Evaluator::evaluate)
        .def("matmul", &Evaluator::matmul)
        .def("enter_scope", &Evaluator::enter_scope)
        .def("exit_scope", &Evaluator::exit_scope);

    // Abstract base classes use trampolines and have NO constructor
    py::class_<nn::Module, PyModule, std::shared_ptr<nn::Module>>(m, "Module")
        .def("forward", &nn::Module::forward)
        .def("parameters", &nn::Module::parameters);

    py::class_<nn::Loss, PyLoss, std::shared_ptr<nn::Loss>>(m, "Loss")
        .def("__call__", &nn::Loss::operator());

    py::class_<nn::Optimizer, PyOptimizer, std::shared_ptr<nn::Optimizer>>(m, "Optimizer")
        .def("zero_grad", &nn::Optimizer::zero_grad)
        .def("step", &nn::Optimizer::step);

    // Concrete classes are bound normally with their constructors
    py::class_<nn::Sequential, nn::Module, std::shared_ptr<nn::Sequential>>(m, "Sequential")
        .def(py::init<>())
        .def("add_module", &nn::Sequential::add_module);

    py::class_<nn::Dense, nn::Module, std::shared_ptr<nn::Dense>>(m, "Dense")
        .def(py::init<long, long>(), py::arg("input_features"), py::arg("output_features"));

    py::class_<nn::ReLU, nn::Module, std::shared_ptr<nn::ReLU>>(m, "ReLU")
        .def(py::init<>());

    py::class_<nn::Sigmoid, nn::Module, std::shared_ptr<nn::Sigmoid>>(m, "Sigmoid")
        .def(py::init<>());

    py::class_<nn::Flatten, nn::Module, std::shared_ptr<nn::Flatten>>(m, "Flatten")
        .def(py::init<>());

    py::class_<nn::MSELoss, nn::Loss, std::shared_ptr<nn::MSELoss>>(m, "MSELoss")
        .def(py::init<>());

    py::class_<nn::CrossEntropyLoss, nn::Loss, std::shared_ptr<nn::CrossEntropyLoss>>(m, "CrossEntropyLoss")
        .def(py::init<>());

    py::class_<nn::SGD, nn::Optimizer, std::shared_ptr<nn::SGD>>(m, "SGD")
        .def(py::init<std::vector<std::shared_ptr<Tensor>>, double>(), py::arg("params"), py::arg("lr") = 0.01);

    py::class_<nn::Adam, nn::Optimizer, std::shared_ptr<nn::Adam>>(m, "Adam")
        .def(py::init<std::vector<std::shared_ptr<Tensor>>, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 0.001, py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999, py::arg("epsilon") = 1e-8);

    py::class_<nn::AssembledModel, std::shared_ptr<nn::AssembledModel>>(m, "AssembledModel")
        .def(py::init<std::shared_ptr<nn::Module>, std::shared_ptr<nn::Optimizer>, std::shared_ptr<nn::Loss>>())
        .def("train", &nn::AssembledModel::train, py::arg("data"), py::arg("labels"), py::arg("epochs") = 10)
        .def_readonly("architecture", &nn::AssembledModel::architecture);
}