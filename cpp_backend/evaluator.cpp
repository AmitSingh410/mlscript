#include "evaluator.hpp"
#include <stdexcept>

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

// Retrieves a variable by searching from the current scope outwards to the global scope.
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

    // Handles mixed types (int, double) by promoting both to double.
    template <typename T, typename U>
    Value operator()(T l, U r) const {
        return (*this)(static_cast<double>(l), static_cast<double>(r));
    }
};

// Evaluates a binary operation using the visitor.
Value Evaluator::evaluate(const std::string& op, const Value& left, const Value& right) {
    return std::visit(OperationVisitor{op}, left, right);
}