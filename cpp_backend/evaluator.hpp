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
#include <memory>
#include <unordered_set>
#include "context.hpp"

namespace py = pybind11;

struct Slice {
    py::ssize_t start;
    py::ssize_t stop;
    py::ssize_t step;
};

class Tensor; 

void build_topo(std::shared_ptr<const Tensor> node, std::vector<std::shared_ptr<const Tensor>>& topo, std::unordered_set<std::shared_ptr<const Tensor>>& visited);

class PYBIND11_EXPORT Tensor : public std::enable_shared_from_this<Tensor> {
public:
    Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> mat;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> grad;

    Tensor(const std::vector<std::vector<double>>& data);
    Tensor(const Eigen::MatrixXd& matrix); 
    Tensor() = default;

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator*(double scalar) const;
    Tensor matmul(const Tensor& other) const;
    Tensor sum() const;

    double get_element(long row, long col) const;
    Tensor get_row(long row) const;
    Tensor slice(Slice row_slice,Slice col_slice) const;

    void backward();
    
    friend void build_topo(std::shared_ptr<const Tensor> node, std::vector<std::shared_ptr<const Tensor>>& topo, std::unordered_set<std::shared_ptr<const Tensor>>& visited);

private:
    std::vector<std::shared_ptr<const Tensor>> _prev;
    std::string _op;
    double _scalar_val = 0.0;
};

Tensor operator*(double scalar, const Tensor& t);

using Value = std::variant<int, double, std::string, bool, py::object>;

class PYBIND11_EXPORT Evaluator {
private:
    std::vector<std::unordered_map<std::string, Value>> scope_stack;

public:
    Evaluator();
    void enter_scope();
    void exit_scope();
    void assign_variable(const std::string& name, const py::object& value);
    void set_grad_enabled(bool enabled);
    py::object get_variable(const std::string& name);
    py::object evaluate(const std::string& op, const py::object& left, const py::object& right);
    py::object matmul(const py::object& left, const py::object& right);
};

#endif