// --- OpenMP Performance Test ---

print("Building a large data structure (1500x1500)...")
// This part is intentionally simple to work with our current language features.
size = 1500
row = []
i = 0
while (i < size) {
    row.append(i)
    i = i + 1
}

data = []
i = 0
while (i < size) {
    data.append(row)
    i = i + 1
}

print("Creating large Tensors...")
a = tensor(data)
b = tensor(data)

print("")
print(">>> Starting parallel matrix multiplication. Watch your CPU monitor now! <<<")

// This is the core operation that should use all available CPU cores.
c = matmul(a, b)

print(">>> Calculation complete. <<<")