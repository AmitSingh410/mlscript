#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

namespace py = pybind11;

struct Slice {
    py::ssize_t start;
    py::ssize_t stop;
    py::ssize_t step;
};

class Tensor {
public:
    Eigen::MatrixXd mat;

    Tensor(const std::vector<std::vector<double>>& data);
    Tensor() = default;

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator*(double scalar) const;
    Tensor matmul(const Tensor& other) const;

    double get_element(long row, long col) const;
    Tensor get_row(long row) const;
    Tensor slice(Slice row_slice,Slice col_slice) const;
};
Tensor operator*(double scalar, const Tensor& t);



using Value = std::variant<int, double, std::string, bool, py::object>;



class Evaluator {
private:
    std::vector<std::unordered_map<std::string, Value>> scope_stack;

public:
    Evaluator();
    void enter_scope();
    void exit_scope();
    void assign_variable(const std::string& name, const py::object& value);
    py::object get_variable(const std::string& name);
    py::object evaluate(const std::string& op, const py::object& left, const py::object& right);
    py::object matmul(const py::object& left, const py::object& right);
};

#endif