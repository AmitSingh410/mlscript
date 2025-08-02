#include "evaluator.hpp"
#include <stdexcept>
#include <cstdlib>

// parse string to double or lookup variable
Value string_to_value(const std::string& s){
    if (s.find('.') != std::string::npos) {
        return std::stod(s);
    } 
    return std::stoi(s);
}
Value Evaluator::eval_expr(const std::string &expr) {
    try {
        return string_to_value(expr);
    } catch (...) {
        if (vars.count(expr)) return vars.at(expr);
        throw std::runtime_error("Undefined variable: " + expr);
    }
}

void Evaluator::set_var(const std::string &name, const Value& val) {
    vars[name] = val;
}

Value Evaluator::get_var(const std::string &name) {
    if (vars.count(name)) return vars.at(name);
    throw std::runtime_error("Undefined variable: " + name);
}

// A visitor for handling binary operations
struct OperationVisitor {
    const std::string& op;
    Value operator()(int l, int r) const {
        if (op == "+") return l + r;
        if (op == "-") return l - r;
        if (op == "*") return l * r;
        if (op == "/") return static_cast<double>(l) / r; // Division results in double
        throw std::runtime_error("Unsupported op for ints: " + op);
    }
    Value operator()(double l, double r) const {
        if (op == "+") return l + r;
        if (op == "-") return l - r;
        if (op == "*") return l * r;
        if (op == "/") return l / r;
        throw std::runtime_error("Unsupported op for doubles: " + op);
    }
    // Handle mixed types by promoting int to double
    template <typename T, typename U>
    Value operator()(T l, U r) const {
        return (*this)(static_cast<double>(l), static_cast<double>(r));
    }
};

Value Evaluator::eval_op(const Value& left, const std::string& op,const Value& right) {
    return std::visit(OperationVisitor{op}, left, right);
}