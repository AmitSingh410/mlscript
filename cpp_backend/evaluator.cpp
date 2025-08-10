#include "evaluator.hpp"
#include <stdexcept>
#include <iostream>

Tensor::Tensor(const std::vector<std::vector<double>>& data) {
    if (data.empty() || data[0].empty()) {
        throw std::runtime_error("Tensor data cannot be empty.");
    }
    size_t rows = data.size();
    size_t cols = data[0].size();
    mat.resize(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        if (data[i].size() != cols) {
            throw std::runtime_error("All rows in tensor data must have the same number of columns.");
        }
        for (size_t j = 0; j < cols; ++j) {
            mat(i, j) = data[i][j];
        }
    }
}

Tensor Tensor::operator+(const Tensor& other) const {
    Tensor result;
    result.mat = this->mat.array() + other.mat.array();
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    Tensor result;
    result.mat = this->mat.array() - other.mat.array();
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    Tensor result;
    result.mat = this->mat.array() * other.mat.array();
    return result;
}

Tensor Tensor::operator/(const Tensor& other) const {
    Tensor result;
    result.mat = this->mat.array() / other.mat.array();
    return result;
}

Tensor Tensor::operator*(double scalar) const {
    Tensor result;
    result.mat = this->mat.array() * scalar;
    return result;
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (this->mat.cols() != other.mat.rows()) {
        throw std::runtime_error("Tensor shapes are incompatible for matrix multiplication.");
    }
    Tensor result;
    result.mat = this->mat * other.mat;
    return result;
}

// Constructor: Initializes the evaluator by creating the global scope.
Evaluator::Evaluator() {
    this->enter_scope();
}

// Enters a new, nested scope by pushing a new symbol table onto the stack.
void Evaluator::enter_scope() {
    scope_stack.emplace_back();
}

// Exits the current scope, ensuring the global scope is never removed.
void Evaluator::exit_scope() {
    if (scope_stack.size() > 1) {
        scope_stack.pop_back();
    } else {
        // This should ideally not be reachable if the interpreter is correct.
        throw std::runtime_error("Internal error: Cannot exit the global scope.");
    }
}

// Assigns a variable to the current, most-nested scope.
void Evaluator::assign_variable(const std::string& name, const Value& value) {
    if (!scope_stack.empty()) {
        scope_stack.back()[name] = value;
    }
}

Value Evaluator::get_variable(const std::string& name) {
    for (auto it = scope_stack.rbegin(); it != scope_stack.rend(); ++it) {
        if (it->count(name)) {
            return it->at(name);
        }
    }
    throw std::runtime_error("Undefined variable: " + name);
}

// A visitor for handling binary operations on std::variant<int, double>.
struct OperationVisitor {
    const std::string& op;

    Value operator()(int l, int r) const {
        if (op == "+") return l + r;
        if (op == "-") return l - r;
        if (op == "*") return l * r;
        if (op == "/") {
            if (r == 0) throw std::runtime_error("Division by zero");
            return static_cast<double>(l) / r; // Division of ints produces a double
        }
        throw std::runtime_error("Unsupported operator for ints: " + op);
    }

    Value operator()(double l, double r) const {
        if (op == "+") return l + r;
        if (op == "-") return l - r;
        if (op == "*") return l * r;
        if (op == "/") {
            if (r == 0) throw std::runtime_error("Division by zero");
            return l / r;
        }
        throw std::runtime_error("Unsupported operator for doubles: " + op);
    }

    Value operator()(const std::string& l, const std::string& r) const {
        if (op == "+") return l + r; // String concatenation
        throw std::runtime_error("Operator '" + op + "' not supported for strings.");
    }

    // Handles mixed types (int, double,string,bool and py objects.
    template <typename T, typename U>
    Value operator()(T l, U r) const {
        if constexpr(std::is_arithmetic_v<T> && std::is_arithmetic_v<U>) {
        return (*this)(static_cast<double>(l), static_cast<double>(r));
    }
        throw std::runtime_error("Unsupported type combination for operator: " + op);
    }
};

// Evaluates a binary operation using the visitor.

Value Evaluator::evaluate(const std::string& op, const Value& left, const Value& right) {
    return std::visit(OperationVisitor{op}, left, right);
}

Tensor operator*(double scalar, const Tensor& t) {
    return t * scalar; 
}

Value Evaluator::evaluate(const std::string& op, const Tensor& left, const Tensor& right) {
    if (op == "+") return left + right;
    if (op == "-") return left - right;
    if (op == "*") return left * right;
    if (op == "/") return left / right;
    throw std::runtime_error("Unsupported operator for Tensors: " + op);
}

Value Evaluator::evaluate(const std::string& op, const Tensor& left, double right) {
    if (op == "*") return left * right;
    throw std::runtime_error("Unsupported operator for Tensor and scalar: " + op);
}

Value Evaluator::evaluate(const std::string& op, double left, const Tensor& right) {
    if (op == "*") return left * right;
    throw std::runtime_error("Unsupported operator for scalar and Tensor: " + op);
}

Value Evaluator::matmul(const Tensor& left, const Tensor& right) {
    return left.matmul(right);
}