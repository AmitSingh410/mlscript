#include "evaluator.hpp"
#include <stdexcept>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <omp.h>
#include <atomic>

void build_topo(const std::shared_ptr<Tensor>& node, std::vector<std::shared_ptr<Tensor>>& topo, std::unordered_set<std::shared_ptr<Tensor>>& visited) {        
    if (visited.find(node) == visited.end()) {                                        
        visited.insert(node);                                                         
        for (const auto& parent_wp : node->_prev) {                                   
            if (auto parent = parent_wp.lock()) {                                     
                build_topo(std::const_pointer_cast<Tensor>(parent), topo, visited);   
            } else {
                throw std::runtime_error("build_topo(): expired parent; node id="+ std::to_string(node->id));
            }
        }
        topo.push_back(node);                                                        
    }
}

static inline void ensure_grad(const std::shared_ptr<Tensor>& t) {             
    if (t->grad.size() == 0) {                                                 
        t->grad = Eigen::MatrixXd::Zero(t->mat.rows(), t->mat.cols());         
    } else if (t->grad.rows() != t->mat.rows() || t->grad.cols() != t->mat.cols()) { 
        t->grad = Eigen::MatrixXd::Zero(t->mat.rows(), t->mat.cols());         
    }
}

// File-scope atomic counter for unique tensor ids
static std::atomic<uint64_t> __tensor_id_counter{1};
/*
Tensor::Tensor(Tensor&& other) noexcept
    : std::enable_shared_from_this<Tensor>(std::move(other)), mat(std::move(other.mat)), grad(std::move(other.grad)),
      _prev(std::move(other._prev)), _op(std::move(other._op)), _scalar_val(other._scalar_val) {
    // This constructor efficiently "steals" the resources from a temporary
    // Tensor object without making unnecessary copies.
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        std::enable_shared_from_this<Tensor>::operator=(std::move(other));
        mat = std::move(other.mat);
        grad = std::move(other.grad);
        _prev = std::move(other._prev);
        _op = std::move(other._op);
        _scalar_val = other._scalar_val;
    }
    return *this;
}
*/
Tensor::Tensor(const Eigen::MatrixXd& matrix)
    : mat(matrix), _op("") {
    grad = Eigen::MatrixXd::Zero(mat.rows(), mat.cols());
    id = __tensor_id_counter.fetch_add(1);
}

Tensor::Tensor(const std::vector<std::vector<double>>& data) : _op("") {
    if (data.empty() || data[0].empty()) {
        mat.resize(0, 0);
        grad.resize(0, 0);
        return;
    }
    ptrdiff_t rows = data.size();
    ptrdiff_t cols = data[0].size();
    mat.resize(rows, cols);
    grad.setZero(rows, cols);
    for (ptrdiff_t i = 0; i < rows; ++i) {
        if (data[i].size() != cols) {
            throw std::runtime_error("All rows in tensor data must have the same number of columns.");
        }
        for (ptrdiff_t j = 0; j < cols; ++j) {
            mat(i, j) = data[i][j];
        }
    }
}

std::shared_ptr<Tensor> Tensor::operator+(const Tensor& other) const {
    return make_binary_op(other, this->mat + other.mat, "+");
}

std::shared_ptr<Tensor> Tensor::operator-(const Tensor& other) const {
    return make_binary_op(other, this->mat - other.mat, "-");
}

std::shared_ptr<Tensor> Tensor::operator*(const Tensor& other) const {
    return make_binary_op(other, this->mat.cwiseProduct(other.mat), "*");
}

std::shared_ptr<Tensor> Tensor::operator/(const Tensor& other) const {
    return make_binary_op(other, this->mat.cwiseQuotient(other.mat), "/");
}

std::shared_ptr<Tensor> Tensor::matmul(const Tensor& other) const {
    return make_binary_op(other, this->mat * other.mat, "matmul");
}

std::shared_ptr<Tensor> Tensor::graph_add(const std::shared_ptr<Tensor>& other) const {
    auto result = std::make_shared<Tensor>(this->mat.array() + other->mat.array());
    if (AutodiffContext::get_instance().is_grad_enabled()) {
        result->_prev = {
    std::weak_ptr<const Tensor>(this->shared_from_this()),
    std::weak_ptr<const Tensor>(other)
    };
        result->_op = "+";

        GraphContext::get_instance().register_tensor(result); 
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::graph_sub(const std::shared_ptr<Tensor>& other) const {
    auto result = std::make_shared<Tensor>(this->mat.array() - other->mat.array());
    if (AutodiffContext::get_instance().is_grad_enabled()) {
        result->_prev = {
    std::weak_ptr<const Tensor>(this->shared_from_this()),
    std::weak_ptr<const Tensor>(other)
    };
        result->_op = "-";

        GraphContext::get_instance().register_tensor(result);
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::graph_mul(const std::shared_ptr<Tensor>& other) const {
    auto result = std::make_shared<Tensor>(this->mat.array() * other->mat.array());
    if (AutodiffContext::get_instance().is_grad_enabled()) {
        result->_prev = {
    std::weak_ptr<const Tensor>(this->shared_from_this()),
    std::weak_ptr<const Tensor>(other)
    };
        result->_op = "*";

        GraphContext::get_instance().register_tensor(result);
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::graph_div(const std::shared_ptr<Tensor>& other) const {
    auto result = std::make_shared<Tensor>(this->mat.array() / other->mat.array());
    if (AutodiffContext::get_instance().is_grad_enabled()) {
        result->_prev = {
    std::weak_ptr<const Tensor>(this->shared_from_this()),
    std::weak_ptr<const Tensor>(other)
    };
        result->_op = "/";

        GraphContext::get_instance().register_tensor(result);
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::graph_matmul(const std::shared_ptr<Tensor>& other) const {
    if (this->mat.cols() != other->mat.rows()) {
        throw std::runtime_error("Tensor shapes are incompatible for matrix multiplication.");
    }
    auto result = std::make_shared<Tensor>(this->mat * other->mat);
    if (AutodiffContext::get_instance().is_grad_enabled()) {
        result->_prev = {
    std::weak_ptr<const Tensor>(this->shared_from_this()),
    std::weak_ptr<const Tensor>(other)
    };
        result->_op = "matmul";

        GraphContext::get_instance().register_tensor(result);
    }
    return result;
}


std::shared_ptr<Tensor> Tensor::graph_mul_scalar(double scalar) const {
    auto result = std::make_shared<Tensor>(this->mat.array() * scalar);
    if (AutodiffContext::get_instance().is_grad_enabled()) {
        result->_prev = { std::weak_ptr<const Tensor>(this->shared_from_this()) };

        result->_op = "*_scalar";
        result->_scalar_val = scalar;

        GraphContext::get_instance().register_tensor(result);
    }
    return result;
}

std::shared_ptr<Tensor> operator*(double scalar, const std::shared_ptr<Tensor>& t) {
    return t->graph_mul_scalar(scalar);
}


std::shared_ptr<Tensor> Tensor::sum() const {
    double scalar_sum = this->mat.sum();
    Eigen::MatrixXd result_mat(1, 1);
    result_mat(0, 0) = scalar_sum;
    
    auto result = std::make_shared<Tensor>(result_mat);
    if (AutodiffContext::get_instance().is_grad_enabled()) {
        result->_prev = { std::weak_ptr<const Tensor>(this->shared_from_this()) };

        result->_op = "sum";

        GraphContext::get_instance().register_tensor(result);
    }
    return result;
}


double Tensor::get_element(long row, long col) const {
    if (row >= mat.rows() || col >= mat.cols() || row < 0 || col < 0) {
        throw std::out_of_range("Tensor index out of range.");
    }
    return mat(row, col);
}

std::shared_ptr<Tensor> Tensor::get_row(long row) const {
    if (row >= mat.rows() || row < 0) {
        throw std::out_of_range("Tensor row index out of range.");
    }
    auto result = std::make_shared<Tensor>(mat.row(row));
    if (AutodiffContext::get_instance().is_grad_enabled()) {
        result->_prev = { std::weak_ptr<const Tensor>(this->shared_from_this()) };

        result->_op = "get_row";
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::slice(Slice row_slice, Slice col_slice) const {
    std::vector<py::ssize_t> row_indices;
    for (py::ssize_t i = row_slice.start; row_slice.step > 0 ? i < row_slice.stop : i > row_slice.stop; i += row_slice.step) {
        row_indices.push_back(i);
    }

    std::vector<py::ssize_t> col_indices;
    for (py::ssize_t i = col_slice.start; col_slice.step > 0 ? i < col_slice.stop : i > col_slice.stop; i += col_slice.step) {
        col_indices.push_back(i);
    }

    if (row_indices.empty() || col_indices.empty()) {
        return std::make_shared<Tensor>(Eigen::MatrixXd(0, 0));
    }
    
    Eigen::MatrixXd new_mat(row_indices.size(), col_indices.size());
    #pragma omp parallel for collapse(2)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(row_indices.size()); ++i) {
        for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(col_indices.size()); ++j) {
            new_mat(i, j) = this->mat(row_indices[i], col_indices[j]);
        }
    }
    
    auto result = std::make_shared<Tensor>(new_mat);
    if (AutodiffContext::get_instance().is_grad_enabled()) {
        result->_prev = { std::weak_ptr<const Tensor>(this->shared_from_this()) };

        result->_op = "slice";
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::softmax() const {
    Eigen::MatrixXd stable_mat = this->mat.rowwise() - this->mat.colwise().maxCoeff();
    Eigen::MatrixXd exp_mat = stable_mat.array().exp();
    Eigen::VectorXd sum_exp = exp_mat.rowwise().sum();
    Eigen::MatrixXd result_mat = exp_mat.array().colwise() / sum_exp.array();

    auto result = std::make_shared<Tensor>(result_mat);
    if (AutodiffContext::get_instance().is_grad_enabled()) {
        result->_prev = { std::weak_ptr<const Tensor>(this->shared_from_this()) };

        result->_op = "softmax";

        GraphContext::get_instance().register_tensor(result);
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::log() const {
    auto result = std::make_shared<Tensor>(this->mat.array().log());
    if (AutodiffContext::get_instance().is_grad_enabled()) {
        result->_prev = { std::weak_ptr<const Tensor>(this->shared_from_this()) };

        result->_op = "log";

        GraphContext::get_instance().register_tensor(result);
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::log_softmax() const {
    Eigen::MatrixXd stable_mat = this->mat;
    
    for (long i = 0; i < stable_mat.rows(); ++i) {
        stable_mat.row(i).array() -= stable_mat.row(i).maxCoeff();
    }

    Eigen::VectorXd log_sum_exp = stable_mat.array().exp().rowwise().sum().log();
    
    for (long i = 0; i < stable_mat.rows(); ++i) {
        stable_mat.row(i).array() -= log_sum_exp(i);
    }

    auto result = std::make_shared<Tensor>(stable_mat);
    if (AutodiffContext::get_instance().is_grad_enabled()) {
        result->_prev = { std::weak_ptr<const Tensor>(this->shared_from_this()) };

        result->_op = "log_softmax";

        GraphContext::get_instance().register_tensor(result);
    }
    return result;
}

void Tensor::backward() {                                                                 
    auto self = shared_from_this();                                                       

    // 1) Build topological order (parents first)
    std::vector<std::shared_ptr<Tensor>> topo;                                          
    std::unordered_set<std::shared_ptr<Tensor>> visited;                               
    build_topo(self, topo, visited);                                                      

    // 2) Seed gradient at the output if empty
    if (grad.size() == 0) {                                                             
        grad = Eigen::MatrixXd::Ones(mat.rows(), mat.cols());                            
    }

    // 3) Backward pass in reverse topological order
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {                             
        auto t = *it;                                                                    
        // No parents -> leaf or value-only tensor
        if (t->_prev.empty()) continue;                                                   

        const auto& op = t->_op;                                                        

        // Utility to fetch parents (up to 2) — expand as needed
        auto P = [&](size_t i) -> std::shared_ptr<Tensor> {                              
            auto p = t->_prev.at(i).lock();                                               
            if (!p) throw std::runtime_error("backward(): expired parent; node id=" + std::to_string(t->id));
            return std::const_pointer_cast<Tensor>(p);                                 
        };

        if (op == "+") {                                                                  
            auto a = P(0), b = P(1);                                                      
            if (a->requires_grad) { ensure_grad(a); a->grad.noalias() += t->grad; }       
            if (b->requires_grad) { ensure_grad(b); b->grad.noalias() += t->grad; }       

        } else if (op == "-") {                                                           
            auto a = P(0), b = P(1);
            if (a->requires_grad) { ensure_grad(a); a->grad.noalias() += t->grad; }       
            if (b->requires_grad) { ensure_grad(b); b->grad.noalias() -= t->grad; }       

        } else if (op == "*") { // elementwise                                           
            auto a = P(0), b = P(1);
            if (a->requires_grad) { ensure_grad(a); a->grad.array() += (t->grad.array() * b->mat.array()); } 
            if (b->requires_grad) { ensure_grad(b); b->grad.array() += (t->grad.array() * a->mat.array()); } 

        } else if (op == "/") { // elementwise                                            
            auto a = P(0), b = P(1);
            if (a->requires_grad) { ensure_grad(a); a->grad.array() += (t->grad.array() / b->mat.array()); } 
            if (b->requires_grad) { ensure_grad(b);                                       
                b->grad.array() += (-t->grad.array() * a->mat.array() / b->mat.array().square()); }
            
        } else if (op == "matmul") {                                                      
            auto A = P(0), B = P(1);
            if (A->requires_grad) { ensure_grad(A); A->grad.noalias() += t->grad * B->mat.transpose(); }     
            if (B->requires_grad) { ensure_grad(B); B->grad.noalias() += A->mat.transpose() * t->grad; }     

        } else if (op == "relu") {                                                       
            auto x = P(0);
            if (x->requires_grad) {
                ensure_grad(x);
                Eigen::ArrayXXd mask = (x->mat.array() > 0.0).cast<double>();             
                x->grad.array() += t->grad.array() * mask;                                
            }

        } else if (op == "sigmoid") {                                                     
            auto x = P(0);
            if (x->requires_grad) {
                ensure_grad(x);
                Eigen::ArrayXXd s = 1.0 / (1.0 + (-x->mat.array()).exp());                
                x->grad.array() += t->grad.array() * (s * (1.0 - s));                     
            }

        } else if (op == "log") {                                                      
            auto x = P(0);
            if (x->requires_grad) {
                ensure_grad(x);
                x->grad.array() += t->grad.array() / x->mat.array();                   
            }

        } else if (op == "sum") { // sum over all elements                              
            auto x = P(0);
            if (x->requires_grad) {
                ensure_grad(x);
                // broadcast scalar grad to x's shape
                double g = (t->grad.size() == 1) ? t->grad(0,0) : t->grad.sum();          
                x->grad.array() += Eigen::ArrayXXd::Constant(x->mat.rows(), x->mat.cols(), g); 
            }

        } else if (op == "flatten") {                                                     
            auto x = P(0);
            if (x->requires_grad) {
                ensure_grad(x);
                Eigen::MatrixXd g = t->grad;                                              
                g.resize(x->mat.rows(), x->mat.cols());                                   
                x->grad.noalias() += g;                                                   
            }

        } else if (op == "log_softmax") {                                                 
            auto x = P(0);
            if (x->requires_grad) {
                ensure_grad(x);
                // softmax = exp(x - max)/sum(exp(...))
                Eigen::MatrixXd shifted = x->mat;                                       
                for (Eigen::Index i=0; i<shifted.rows(); ++i) shifted.row(i).array() -= shifted.row(i).maxCoeff();
                Eigen::MatrixXd expm = shifted.array().exp();                             
                Eigen::VectorXd sumexp = expm.rowwise().sum();                            
                Eigen::MatrixXd sm = expm.array().colwise() / sumexp.array();             
                Eigen::VectorXd gsum = t->grad.rowwise().sum();                           
                Eigen::MatrixXd gsum_rep = gsum.replicate(1, sm.cols());                  
                x->grad.array() += t->grad.array() - (sm.array() * gsum_rep.array());     
            } 
        } else if (op == "*_scalar") {
            auto x = P(0);
            if (x->requires_grad) {
                ensure_grad(x);
                x->grad.noalias() += t->grad * t->_scalar_val;
            }

        } else {
            throw std::runtime_error("backward(): unknown op '" + op + "', node id=" + std::to_string(t->id)); 
        }
    }
}


Evaluator::Evaluator() {
    this->enter_scope();
}

void Evaluator::enter_scope() {
    scope_stack.emplace_back();
}

void Evaluator::exit_scope() {
    if (scope_stack.size() > 1) {
        scope_stack.pop_back();
    } else {
        throw std::runtime_error("Internal error: Cannot exit the global scope.");
    }
}

void Evaluator::assign_variable(const std::string& name, const py::object& value) {
    if (!scope_stack.empty()) {
        scope_stack.back()[name] = value;
    }
}

void Evaluator::set_grad_enabled(bool enabled) {
    AutodiffContext::get_instance().set_grad_enabled(enabled);
}

py::object Evaluator::get_variable(const std::string& name) {
    for (auto it = scope_stack.rbegin(); it != scope_stack.rend(); ++it) {
        if (it->count(name)) {
            const Value& val = it->at(name);
            if (std::holds_alternative<py::object>(val)) {
                return std::get<py::object>(val);
            } else if (std::holds_alternative<int>(val)) {
                return py::cast(std::get<int>(val));
            } else if (std::holds_alternative<double>(val)) {
                return py::cast(std::get<double>(val));
            } else if (std::holds_alternative<std::string>(val)) {
                return py::cast(std::get<std::string>(val));
            } else if (std::holds_alternative<bool>(val)) {
                return py::cast(std::get<bool>(val));
            }
        }
    }
    throw std::runtime_error("Undefined variable: " + name);
}

py::object Evaluator::evaluate(const std::string& op, const py::object& left, const py::object& right) {
    bool is_left_tensor = py::isinstance<Tensor>(left);
    bool is_left_array = py::isinstance<py::array>(left);
    bool is_left_numeric = py::isinstance<py::float_>(left) || py::isinstance<py::int_>(left);

    bool is_right_tensor = py::isinstance<Tensor>(right);
    bool is_right_array = py::isinstance<py::array>(right);
    bool is_right_numeric = py::isinstance<py::float_>(right) || py::isinstance<py::int_>(right);

    const char* op_dunder = nullptr;
    if (op == "+") op_dunder = "__add__";
    else if (op == "-") op_dunder = "__sub__";
    else if (op == "*") op_dunder = "__mul__";
    else if (op == "/") op_dunder = "__truediv__";

    if (op_dunder) {
        if (is_left_array && is_right_tensor) {
            return left.attr(op_dunder)(py::cast(right.cast<const Tensor&>()));
        }
        if (is_left_tensor && is_right_array) {
            return py::cast(left.cast<const Tensor&>()).attr(op_dunder)(right);
        }
        if (is_left_array && is_right_numeric) {
             return left.attr(op_dunder)(right);
        }
        if (is_left_numeric && is_right_array) {
            std::string r_op_dunder = "__r" + std::string(op_dunder + 2);
            return right.attr(r_op_dunder.c_str())(left);
        }
    }

    if (is_left_tensor && is_right_tensor) {
        auto l = left.cast<std::shared_ptr<Tensor>>();
        auto r = right.cast<std::shared_ptr<Tensor>>();
        if (op == "+") return py::cast(l->graph_add(r));
        if (op == "-") return py::cast(l->graph_sub(r));
        if (op == "*") return py::cast(l->graph_mul(r));
        if (op == "/") return py::cast(l->graph_div(r));
    }
    // Handle Tensor * scalar
    if (is_left_tensor && is_right_numeric) {
        if (op == "*") {
            auto l_ptr = left.cast<std::shared_ptr<Tensor>>();
            double r_scalar = right.cast<double>();
            return py::cast(l_ptr->graph_mul_scalar(r_scalar));
        }
    }
    // Handle scalar * Tensor
    if (is_left_numeric && is_right_tensor) {
        if (op == "*") {
            double l_scalar = left.cast<double>();
            auto r_ptr = right.cast<std::shared_ptr<Tensor>>();
            return py::cast(operator*(l_scalar, r_ptr));
        }
    }
    
    if ((is_left_numeric && py::isinstance<py::float_>(right)) || (py::isinstance<py::float_>(left) && is_right_numeric) || (py::isinstance<py::float_>(left) && py::isinstance<py::float_>(right))) {
        double l = left.cast<double>();
        double r = right.cast<double>();
        if (op == "+") return py::cast(l + r);
        if (op == "-") return py::cast(l - r);
        if (op == "*") return py::cast(l * r);
        if (op == "/") return py::cast(l / r);
    }
    if (py::isinstance<py::int_>(left) && py::isinstance<py::int_>(right)) {
        int l = left.cast<int>();
        int r = right.cast<int>();
        if (op == "+") return py::cast(l + r);
        if (op == "-") return py::cast(l - r);
        if (op == "*") return py::cast(l * r);
        if (op == "/") return py::cast(static_cast<double>(l) / r);
    }
    if (py::isinstance<py::str>(left) && py::isinstance<py::str>(right)) {
        if (op == "+") return py::cast(left.cast<std::string>() + right.cast<std::string>());
    }
    
    std::string left_type = py::str(left.get_type().attr("__name__"));
    std::string right_type = py::str(right.get_type().attr("__name__"));
    throw std::runtime_error("Unsupported types for operator " + op + ": '" + left_type + "' and '" + right_type + "'");
}

py::object Evaluator::matmul(const py::object& left, const py::object& right) {
    if (py::isinstance<Tensor>(left) && py::isinstance<Tensor>(right)) {
        auto l = left.cast<std::shared_ptr<Tensor>>();
        auto r = right.cast<std::shared_ptr<Tensor>>();
        return py::cast(l->graph_matmul(r));
    }
    throw std::runtime_error("matmul is only defined for Tensors.");
}

GraphContext& GraphContext::get_instance() {
    static GraphContext instance;
    return instance;
}

void GraphContext::clear_tape() {
    tape_.clear();
}

void GraphContext::register_tensor(std::shared_ptr<Tensor> t) {
    tape_.push_back(t);
}
