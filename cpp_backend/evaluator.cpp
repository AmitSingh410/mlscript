#include "evaluator.hpp"
#include <stdexcept>
#include <cstdlib>

// parse string to double or lookup variable
double Evaluator::eval_expr(const std::string &expr) {
    try {
        return std::stod(expr);
    } catch (...) {
        if (vars.count(expr)) return vars[expr];
        throw std::runtime_error("Undefined variable: " + expr);
    }
}

void Evaluator::set_var(const std::string &name, const std::string &expr) {
    double v = eval_expr(expr);
    vars[name] = v;
}

double Evaluator::get_var(const std::string &name) {
    if (vars.count(name)) return vars[name];
    throw std::runtime_error("Undefined variable: " + name);
}

double Evaluator::eval_op(const std::string &l, const std::string &op, const std::string &r) {
    double lhs = eval_expr(l);
    double rhs = eval_expr(r);
    if (op == "+") return lhs + rhs;
    if (op == "-") return lhs - rhs;
    if (op == "*") return lhs * rhs;
    if (op == "/") return lhs / rhs;
    throw std::runtime_error("Unsupported op: " + op);
}