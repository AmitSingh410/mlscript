#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <stdexcept>

/**
 * Core C++ evaluator: stores variables in a scoped symbol table and evaluates expressions.
 */
using Value = std::variant<int, double, std::string>;

class Evaluator {
private:
    // A stack of scopes to handle local variables in functions.
    // The last element is the current, most-nested scope.
    std::vector<std::unordered_map<std::string, Value>> scope_stack;

public:
    // Constructor initializes the global scope.
    Evaluator();

    // Enters a new, nested scope (e.g., for a function call).
    void enter_scope();

    // Exits the current scope and returns to the parent scope.
    void exit_scope();

    // Assigns a value to a variable in the current scope.
    void assign_variable(const std::string& name, const Value& value);

    // Retrieves a variable's value, searching from the current scope outwards.
    Value get_variable(const std::string& name);

    // Evaluates a binary operation: left op right.
    Value evaluate(const std::string& op, const Value& left, const Value& right);
};

#endif // EVALUATOR_HPP