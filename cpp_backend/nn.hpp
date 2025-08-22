#ifndef NN_HPP
#define NN_HPP

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <cmath>
#include "evaluator.hpp" // Assumes this includes Eigen and your Tensor class

namespace nn {

// Forward Declarations
class Module;
class Optimizer;
class Loss;

// ===================================================================
// I. CORE ARCHITECTURAL COMPONENTS
// ===================================================================

/**
 * @brief Base class for all neural network modules (layers, activations, etc.).
 */
class Module {
public:
    virtual ~Module() = default;
    virtual std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) = 0;
    virtual std::vector<std::shared_ptr<Tensor>> parameters() {
        return {};
    }
};

/**
 * @brief A container for stacking modules sequentially.
 */
class Sequential : public Module {
private:
    std::vector<std::shared_ptr<Module>> layers;

public:
    void add_module(std::shared_ptr<Module> module) {
        layers.push_back(module);
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override {
        auto current_output_ptr = input;
        for (const auto& layer : layers) {
            current_output_ptr = layer->forward(current_output_ptr);
        }
        return current_output_ptr;
    }

    std::vector<std::shared_ptr<Tensor>> parameters() override {
        std::vector<std::shared_ptr<Tensor>> all_params;
        for (const auto& layer : layers) {
            auto layer_params = layer->parameters();
            all_params.insert(all_params.end(), layer_params.begin(), layer_params.end());
        }
        return all_params;
    }
};

// ===================================================================
// II. STANDARD LAYERS AND ACTIVATIONS
// ===================================================================

/**
 * @brief A standard fully connected (dense) layer.
 */
class Dense : public Module {
private:
    std::shared_ptr<Tensor> weights;
    std::shared_ptr<Tensor> biases;

public:
    Dense(long input_features, long output_features) {
        weights = std::make_shared<Tensor>(Eigen::MatrixXd::Random(input_features, output_features));
        biases = std::make_shared<Tensor>(Eigen::MatrixXd::Random(1, output_features));
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override {
        auto matmul_result = input->graph_matmul(weights);
        auto add_result = matmul_result->graph_add(biases);
        return add_result;
    }

    std::vector<std::shared_ptr<Tensor>> parameters() override {
        return {weights, biases};
    }
};

/**
 * @brief Rectified Linear Unit (ReLU) activation function.
 */
class ReLU : public Module {
public:
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override {
        auto result = std::make_shared<Tensor>(input->mat.array().max(0));
        if (AutodiffContext::get_instance().is_grad_enabled()) {
            result->_prev = {input};
            result->_op = "relu";

            GraphContext::get_instance().register_tensor(result);
        }
        return result;
    }
};

/**
 * @brief Sigmoid activation function.
 */
class Sigmoid : public Module {
public:
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override {
        auto result = std::make_shared<Tensor>(1.0 / (1.0 + (-input->mat.array()).exp()));
        if (AutodiffContext::get_instance().is_grad_enabled()) {
             result->_prev = {input};
             result->_op = "sigmoid"; // You would need to add this backprop rule to evaluator.cpp
        
            GraphContext::get_instance().register_tensor(result);    
        }
        return result;
    }
};

/**
 * @brief Flattens a tensor into a 1D vector (1xN matrix).
 */
class Flatten : public Module {
public:
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input) override {
        Eigen::MatrixXd mat = input->mat;
        mat.resize(1, input->mat.rows() * input->mat.cols());
        
        auto result = std::make_shared<Tensor>(mat);
        if (AutodiffContext::get_instance().is_grad_enabled()) {
            result->_prev = {input};
            result->_op = "flatten";

            GraphContext::get_instance().register_tensor(result);
        }
        return result;
    }
};

// ===================================================================
// III. LOSS FUNCTIONS
// ===================================================================

/**
 * @brief Base class for all loss functions.
 */
class Loss {
public:
    virtual ~Loss() = default;
    virtual std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& predictions, const std::shared_ptr<Tensor>& targets) = 0;
};

/**
 * @brief Mean Squared Error loss.
 */
class MSELoss : public Loss {
public:
    std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& predictions, const std::shared_ptr<Tensor>& targets) override {
        auto diff = predictions->graph_sub(targets);
        auto squared_diff = diff->graph_mul(diff);
        return squared_diff;
    }
};

/**
 * @brief Cross-Entropy Loss (placeholder implementation).
 */
class CrossEntropyLoss : public Loss {
public:
    std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& predictions, const std::shared_ptr<Tensor>& targets) override {
        auto log_probs = predictions->log_softmax();
        auto nll = log_probs->graph_mul(targets);
        auto total_nll = nll->sum();
        auto scaled_nll = total_nll->graph_mul_scalar(-1.0);
        return scaled_nll;
    }
};

// ===================================================================
// IV. OPTIMIZERS
// ===================================================================

/**
 * @brief Base class for all optimizers.
 */
class Optimizer {
protected:
    std::vector<std::shared_ptr<Tensor>> params;
    double learning_rate;

public:
    Optimizer(std::vector<std::shared_ptr<Tensor>> params, double lr)
        : params(params), learning_rate(lr) {}
    virtual ~Optimizer() = default;

    void zero_grad() {
        for (auto& p : params) {
            p->grad.setZero();
        }
    }
    virtual void step() = 0;
};

/**
 * @brief Stochastic Gradient Descent (SGD) optimizer.
 */
class SGD : public Optimizer {
public:
    using Optimizer::Optimizer;
    void step() override {
        for (auto& p : params) {
            p->mat -= learning_rate * p->grad;
        }
    }
};

/**
 * @brief Adam optimizer.
 */
class Adam : public Optimizer {
private:
    std::vector<Eigen::MatrixXd> m;
    std::vector<Eigen::MatrixXd> v;
    double beta1, beta2, epsilon;
    int t;

public:
    Adam(std::vector<std::shared_ptr<Tensor>> params, double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
        : Optimizer(params, lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {
        for (const auto& p : params) {
            m.push_back(Eigen::MatrixXd::Zero(p->mat.rows(), p->mat.cols()));
            v.push_back(Eigen::MatrixXd::Zero(p->mat.rows(), p->mat.cols()));
        }
    }

    void step() override {
        t++;
        for (size_t i = 0; i < params.size(); ++i) {
            m[i] = beta1 * m[i] + (1 - beta1) * params[i]->grad;
            v[i] = beta2 * v[i] + (1 - beta2) * params[i]->grad.array().square().matrix();
            Eigen::MatrixXd m_hat = m[i] / (1 - std::pow(beta1, t));
            Eigen::MatrixXd v_hat = v[i] / (1 - std::pow(beta2, t));
            params[i]->mat -= (learning_rate * m_hat.array() / (v_hat.array().sqrt() + epsilon)).matrix();
        }
    }
};

// ===================================================================
// V. THE ASSEMBLED MODEL
// ===================================================================

/**
 * @brief High-level container that bundles a full training setup.
 */
class AssembledModel {
public:
    std::shared_ptr<nn::Module> architecture;
    std::shared_ptr<nn::Optimizer> optimizer;
    std::shared_ptr<nn::Loss> loss_fn;

    AssembledModel(
        std::shared_ptr<nn::Module> arch,
        std::shared_ptr<nn::Optimizer> opt,
        std::shared_ptr<nn::Loss> loss
    ) : architecture(arch), optimizer(opt), loss_fn(loss) {}

    void train(const std::shared_ptr<Tensor>& data, const std::shared_ptr<Tensor>& labels, int epochs) {
        std::cout << "[mlscript] AssembledModel starting training..." << std::endl;
        for (int i = 0; i < epochs; ++i) {
            
            GraphContext::get_instance().clear_tape();
            optimizer->zero_grad();
            
            auto predictions = this->architecture->forward(data);
            auto loss = (*this->loss_fn)(predictions, labels);
            
            loss->backward();
            optimizer->step();

            int epoch_tenth = epochs/10;
            if(i % epoch_tenth == 0){
                std::cout << "  Epoch " << i + 1 << " complete. Loss: " << loss->get_element(0, 0) << std::endl;
            }
        }
        GraphContext::get_instance().clear_tape();
        std::cout << "[mlscript] Training complete." << std::endl;
    }
    
};

} // namespace nn

#endif // NN_HPP