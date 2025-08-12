#ifndef AUTODIFF_CONTEXT_HPP
#define AUTODIFF_CONTEXT_HPP

class AutodiffContext {
public:
    static AutodiffContext& get_instance() {
        static AutodiffContext instance;
        return instance;
    }

    void set_grad_enabled(bool enabled) {
        grad_enabled_ = enabled;
    }

    bool is_grad_enabled() const {
        return grad_enabled_;
    }

    AutodiffContext(const AutodiffContext&) = delete;
    void operator=(const AutodiffContext&) = delete;

private:
    AutodiffContext() : grad_enabled_(true) {}
    bool grad_enabled_;
};

#endif // AUTODIFF_CONTEXT_HPP

