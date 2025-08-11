#include "evaluator.hpp"
#include <stdexcept>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

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

Tensor::Tensor(const Eigen::MatrixXd& matrix) : mat(matrix) {}

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
void Evaluator::assign_variable(const std::string& name, const py::object& value) {
    if (!scope_stack.empty()) {
        scope_stack.back()[name] = value;
    }
}

py::object Evaluator::get_variable(const std::string& name) {
    for (auto it = scope_stack.rbegin(); it != scope_stack.rend(); ++it) {
        if (it->count(name)) {
            // All complex objects are stored as py::object, so we get that.
            // Primitives will need to be handled or cast as needed.
            // For now, this direct return is the goal of the refactor.
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

// Evaluates an operation between two objects, which can be Tensors or primitive types.
// This function is designed to handle both Tensor operations and primitive type operations.
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

    // Handle operations involving numpy arrays by delegating back to Python
    if (op_dunder) {
        // Case 1: numpy_array (+, -, *, /) Tensor
        if (is_left_array && is_right_tensor) {
            // Convert our Tensor to a numpy array view and then let Python handle it.
            return left.attr(op_dunder)(py::cast(right.cast<const Tensor&>()));
        }
        // Case 2: Tensor (+, -, *, /) numpy_array
        if (is_left_tensor && is_right_array) {
            return py::cast(left.cast<const Tensor&>()).attr(op_dunder)(right);
        }
        // Case 3: numpy_array (+, -, *, /) scalar
        if (is_left_array && is_right_numeric) {
             return left.attr(op_dunder)(right);
        }
        // Case 4: scalar (+, -, *, /) numpy_array
        if (is_left_numeric && is_right_array) {
            // For scalar on the left, we need the "reflected" operator (e.g., __radd__)
            std::string r_op_dunder = "__r" + std::string(op_dunder + 2);
            return right.attr(r_op_dunder.c_str())(left);
        }
    }

    // --- Original Logic for mlscript Native Types ---

    // Tensor + Tensor
    if (is_left_tensor && is_right_tensor) {
        const auto& l = left.cast<const Tensor&>();
        const auto& r = right.cast<const Tensor&>();
        if (op == "+") return py::cast(l + r);
        if (op == "-") return py::cast(l - r);
        if (op == "*") return py::cast(l * r);
        if (op == "/") return py::cast(l / r);
    }
    // Tensor * scalar (Broadcasting)
    if (is_left_tensor && is_right_numeric) {
        if (op == "*") return py::cast(left.cast<const Tensor&>() * right.cast<double>());
    }
    // scalar * Tensor (Broadcasting)
    if (is_left_numeric && is_right_tensor) {
        if (op == "*") return py::cast(left.cast<double>() * right.cast<const Tensor&>());
    }
    // Primitive arithmetic
    if (is_left_numeric && py::isinstance<py::float_>(right) || py::isinstance<py::float_>(left)) {
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
    // String concatenation
    if (py::isinstance<py::str>(left) && py::isinstance<py::str>(right)) {
        if (op == "+") return py::cast(left.cast<std::string>() + right.cast<std::string>());
    }
    
    throw std::runtime_error("Unsupported types for operator " + op);
}


Tensor operator*(double scalar, const Tensor& t) {
    return t * scalar; 
}


py::object Evaluator::matmul(const py::object& left, const py::object& right) {
    if (py::isinstance<Tensor>(left) && py::isinstance<Tensor>(right)) {
        return py::cast(left.cast<const Tensor&>().matmul(right.cast<const Tensor&>()));
    }
    throw std::runtime_error("matmul is only defined for Tensors.");
}

double Tensor::get_element(long row, long col) const {
    if (row >= mat.rows() || col >= mat.cols() || row < 0 || col < 0) {
        throw std::out_of_range("Tensor index out of range.");
    }
    return mat(row, col);
}

Tensor Tensor::get_row(long row) const {
    if (row >= mat.rows() || row < 0) {
        throw std::out_of_range("Tensor row index out of range.");
    }
    Tensor result;
    result.mat = mat.row(row);
    return result;
}

Tensor Tensor::slice(Slice row_slice, Slice col_slice) const {
    std::vector<py::ssize_t> row_indices;
    for (py::ssize_t i = row_slice.start; row_slice.step > 0 ? i < row_slice.stop : i > row_slice.stop; i += row_slice.step) {
        row_indices.push_back(i);
    }

    std::vector<py::ssize_t> col_indices;
    for (py::ssize_t i = col_slice.start; col_slice.step > 0 ? i < col_slice.stop : i > col_slice.stop; i += col_slice.step) {
        col_indices.push_back(i);
    }

    if (row_indices.empty() || col_indices.empty()) {
        // Return an empty tensor if the slice is empty
        return Tensor(std::vector<std::vector<double>>());
    }

    Tensor result;
    result.mat.resize(row_indices.size(), col_indices.size());
    for(size_t i = 0; i < row_indices.size(); ++i) {
        for(size_t j = 0; j < col_indices.size(); ++j) {
            result.mat(i, j) = this->mat(row_indices[i], col_indices[j]);
        }
    }
    return result;
}