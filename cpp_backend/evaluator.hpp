#pragma once
#include <string>
#include <unordered_map>
#include <variant>
/**
 * Core C++ evaluator: stores variables and evaluates expressions.
 */

using Value = std::variant<int, double>;
class Evaluator {
public:
    // variable storage
    std::unordered_map<std::string,Value> vars;

    // evaluate a single token or variable name
    Value eval_expr(const std::string &expr);
    
    // assign var = expr
    void set_var(const std::string &name, const Value& val);
    
    // get variable value
    Value get_var(const std::string &name);
    
    // evaluate binary operation: left op right
    Value eval_op(const Value& left, const std::string &op, const Value& right);
};