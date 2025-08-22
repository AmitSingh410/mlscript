import "time" as time

print("--- MLScript Deterministic NN Test ---")
print("1. Generating constant, predictable data...")

// --- Create a constant, non-random dataset ---
// We'll create a simple 3-class classification problem.
// Class 0: First element is high.
// Class 1: Second element is high.
// Class 2: Third element is high.

X_list = [
    // Class 0 samples
    [0.9, 0.1, 0.1],
    [1.0, 0.2, 0.1],
    [0.8, 0.1, 0.2],
    // Class 1 samples
    [0.1, 0.9, 0.1],
    [0.2, 1.0, 0.1],
    [0.1, 0.8, 0.2],
    // Class 2 samples
    [0.1, 0.1, 0.9],
    [0.2, 0.1, 1.0],
    [0.1, 0.2, 0.8]
]

// Create corresponding one-hot encoded labels
y_list = [
    // Class 0 labels
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    // Class 1 labels
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    // Class 2 labels
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0]
]

X_train = tensor(X_list)
y_train = tensor(y_list)

print("   Data generation complete.")

// --- Model Definition (Updated for new data shape) ---
print("2. Assembling model from blueprint...")
model = network {
    input: 3, // 3 input features
    layers: [
        Dense(8), // Hidden layer with 8 neurons
        ReLU(),
        Dense(3)  // 3 output classes
    ],
    optimizer: Adam(lr=0.001), // A slightly higher learning rate for a simple problem
    loss: CrossEntropyLoss()
}
print("learning rate=0.001")
print("   Model assembled successfully.")


// --- Training & Performance Measurement ---
print("\n3. Starting model training...")
start_time = time.time()

model.train(X_train, y_train, epochs=150000)

end_time = time.time()
duration = end_time - start_time
print("--- Training Finished ---")
print("   Total training time: ", duration, "seconds")


// --- Performing inference test ---
print("\n4. Performing inference test on the first sample...")
// The first sample belongs to Class 0
test_sample = X_train[0]
prediction = model.architecture.forward(test_sample)

print("   Model prediction (output logits):")
print(prediction)
print("\n--- Test Complete ---")