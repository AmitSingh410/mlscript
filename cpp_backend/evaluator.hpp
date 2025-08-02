#pragma once
#include <string>
#include <unordered_map>

/**
 * Core C++ evaluator: stores variables and evaluates expressions.
 */
class Evaluator {
public:
    // variable storage
    std::unordered_map<std::string,double> vars;

    // evaluate a single token or variable name
    double eval_expr(const std::string &expr);
    
    // assign var = expr
    void set_var(const std::string &name, const std::string &expr);
    
    // get variable value
    double get_var(const std::string &name);
    
    // evaluate binary operation: left op right
    double eval_op(const std::string &left, const std::string &op, const std::string &right);
};