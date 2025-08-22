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

namespace nn{
    class ReLU;
    class Sigmoid;
    class Flatten;
}

class Tensor; 

void build_topo(const std::shared_ptr<Tensor>& node, std::vector<std::shared_ptr<Tensor>>& topo, std::unordered_set<std::shared_ptr<Tensor>>& visited);


class PYBIND11_EXPORT Tensor : public std::enable_shared_from_this<Tensor> {
public:
    Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> mat;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> grad;

    Tensor(const Tensor& other) = delete; // Copy Constructor
    Tensor& operator=(const Tensor& other) = delete; // Copy Assignment Operator
    Tensor(Tensor&& other) noexcept = default; // Move Constructor
    Tensor& operator=(Tensor&& other) noexcept = default;
    
    // Factory helper to ensure Tensor instances are created as shared_ptrs and set requires_grad explicitly.
    static std::shared_ptr<Tensor> create(const Eigen::MatrixXd& m, bool requires_grad = false) {  // (5)
    auto t = std::make_shared<Tensor>(m);                                                      // (6)
    t->requires_grad = requires_grad;                                                          // (7)
    return t;                                                                                  // (8)
    }

    Tensor(const std::vector<std::vector<double>>& data);
    Tensor(const Eigen::MatrixXd& matrix); 
    Tensor() = default;

    std::shared_ptr<Tensor> operator+(const Tensor& other) const;
    std::shared_ptr<Tensor> operator-(const Tensor& other) const;
    std::shared_ptr<Tensor> operator*(const Tensor& other) const;
    std::shared_ptr<Tensor> operator/(const Tensor& other) const;
    std::shared_ptr<Tensor> matmul(const Tensor& other) const;

    std::shared_ptr<Tensor> graph_add(const std::shared_ptr<Tensor>& other) const;
    std::shared_ptr<Tensor> graph_sub(const std::shared_ptr<Tensor>& other) const;
    std::shared_ptr<Tensor> graph_mul(const std::shared_ptr<Tensor>& other) const;
    std::shared_ptr<Tensor> graph_div(const std::shared_ptr<Tensor>& other) const;
    std::shared_ptr<Tensor> graph_mul_scalar(double scalar) const;
    std::shared_ptr<Tensor> graph_matmul(const std::shared_ptr<Tensor>& other) const;
    std::shared_ptr<Tensor> sum() const;
    std::shared_ptr<Tensor> softmax() const;
    std::shared_ptr<Tensor> log() const;
    std::shared_ptr<Tensor> log_softmax() const;
    

    double get_element(long row, long col) const;
    std::shared_ptr<Tensor> get_row(long row) const;
    std::shared_ptr<Tensor> slice(Slice row_slice,Slice col_slice) const;

    void backward();
    void zero_grad() {
        this->grad.setZero();
    }
    
    friend void ::build_topo(const std::shared_ptr<Tensor>& node, std::vector<std::shared_ptr<Tensor>>& topo, std::unordered_set<std::shared_ptr<Tensor>>& visited);
    friend class nn::ReLU;
    friend class nn::Sigmoid;
    friend class nn::Flatten;

    static std::shared_ptr<Tensor> random(long rows, long cols) {
        return std::make_shared<Tensor>(Eigen::MatrixXd::Random(rows,cols));
    }

private:
    std::vector<std::weak_ptr<const Tensor>> _prev;
    std::string _op;
    double _scalar_val = 0.0;

    // Flag: whether this tensor participates in the autograd graph tree.
    bool requires_grad = false;
    // Unique id for tracing ownership and lifetime issues.
    uint64_t id = 0; 

    std::shared_ptr<Tensor> make_binary_op(const Tensor& other, const Eigen::MatrixXd& result, const std::string& op) const 
{
    auto out = std::make_shared<Tensor>(result);

    if (auto self_ptr = this->shared_from_this())
        out->_prev.push_back(self_ptr);
    if (auto other_ptr = other.shared_from_this())
        out->_prev.push_back(other_ptr);

    out->_op = op;
    return out;
}

};

std::shared_ptr<Tensor> operator*(double scalar, const std::shared_ptr<Tensor>& t);

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

class GraphContext {
    public:
        static GraphContext& get_instance();

        void clear_tape();
        void register_tensor(std::shared_ptr<Tensor> t);
        
        GraphContext(const GraphContext&) = delete;
        void operator=(const GraphContext&) = delete;

    private:
        GraphContext() = default;
        std::vector<std::shared_ptr<Tensor>> tape_;

};

#endif