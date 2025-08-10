// ######################################################
// ## mlscript v0.4 - Comprehensive Test Suite
// ######################################################

print("--- 1. Testing Primitives, Variables, and Operators ---")

// Test variable assignment and basic types
a = 10
b = 3.14
c = "hello"
d = true

print("Integer a:")
print(a)
print("Float b:")
print(b)
print("String c:")
print(c)
print("Boolean d:")
print(d)

// Test arithmetic and string concatenation
sum_val = a + 5
product = b * 2
greeting = c + " mlscript!"
print("Sum (10 + 5):")
print(sum_val)
print("Product (3.14 * 2):")
print(product)
print("Greeting:")
print(greeting)

print("") // Print a blank line for spacing

// --- 2. Testing Data Structures (Lists) and Built-ins ---

print("--- 2. Testing Data Structures and Built-in Functions ---")

empty_list = []
data_list = [10, "world", false, 3.14]

print("Empty list:")
print(empty_list)
print("List with mixed types:")
print(data_list)

string_len = len("hello world")
list_len = len(data_list)

print("Length of string 'hello world':")
print(string_len) // Expected: 11
print("Length of data_list:")
print(list_len)   // Expected: 4

print("")

// --- 3. Testing Control Flow ---

print("--- 3. Testing Control Flow (if, while, for) ---")

test_val = 20
if (test_val > 10) {
    print("If block: test_val is greater than 10.")
} else {
    print("If block: test_val is NOT greater than 10.")
}

// Test elif and nested logic (simulating 'and')
if (test_val == 0) {
    print("Elif block: test_val is 0.")
} elif (test_val > 10) {
    if (test_val == 20) {
        print("Elif/Nested block: test_val is exactly 20.")
    }
}

// Test while loop
print("While loop countdown:")
counter = 3
while (counter > 0) {
    print(counter)
    counter = counter - 1
}

// Test for loop with range()
print("For loop with range(5):")
for i in range(5) {
    print(i)
}

// Test for loop over a list
print("For loop over a list [100, 200, 300]:")
loop_list = [100, 200, 300]
for item in loop_list {
    print(item)
}

print("")

// --- 4. Testing Functions and Scope ---

print("--- 4. Testing Functions, Scope, and Return ---")

fun add(x, y) {
    return x + y
}

fun scope_test(x) {
    y = 100 // This 'y' is local to the function
    print("Inside scope_test, local y is:")
    print(y)
    return x * 2
}

fun factorial(n) {
    if (n == 0) {
        return 1
    }
    return n * factorial(n - 1)
}


result_add = add(15, 7)
print("Result of add(15, 7):")
print(result_add) // Expected: 22

y = 5 // Global 'y'
print("Before scope_test, global y is:")
print(y)

scope_test(10)

print("After scope_test, global y is still:")
print(y) // Expected: 5

fact_5 = factorial(5)
print("Result of factorial(5):")
print(fact_5) // Expected: 120

print("")

// --- 5. Demonstrating v0.3 Runtime Errors ---
print("--- 5. Demonstrating v0.3 Runtime Errors ---")
print("The following lines are commented out to prevent crashing the script.")
// print(undefined_variable)
// some_function_that_does_not_exist()

print("")

// ######################################################
// ## v0.4 - NEW FEATURE TESTS
// ######################################################

print("--- 6. Testing v0.4 Interactive Data Structures ---")

// Test List Indexing and Assignment
print("Testing List Indexing and Assignment:")
numbers = [10, 20, 30, 40]
print("Original list:")
print(numbers)
print("Accessing numbers[2]:")
print(numbers[2]) // Expected: 30
numbers[0] = 99
print("After numbers[0] = 99, list is:")
print(numbers) // Expected: [99, 20, 30, 40]
print("")

// Test Dictionary Literals, Access, and Assignment
print("Testing Dictionaries:")
person = {"name": "mlscript", "version": 0.4, "stable": false}
print("Original dictionary:")
print(person)
print("Accessing person[\"name\"]:")
print(person["name"]) // Expected: "mlscript"

// Update existing key
person["stable"] = true
// Add new key
person["year"] = 2025
print("After updates, dictionary is:")
print(person) // Expected: {'name': 'mlscript', 'version': 0.4, 'stable': true, 'year': 2025}
print("")

// Test String Indexing
print("Testing String Indexing:")
lang = "python"
print("Accessing lang[0]:")
print(lang[0]) // Expected: "p"
print("")

// Test new built-in functions
print("Testing New Built-in Functions (min, max, sum):")
scores = [88, 95, 72, 100, 81]
print("Scores:")
print(scores)
print("min(scores):")
print(min(scores)) // Expected: 72
print("max(scores):")
print(max(scores)) // Expected: 100
print("sum(scores):")
print(sum(scores)) // Expected: 436
print("")


// ######################################################
// ## v0.5 - NEW FEATURE TESTS
// ######################################################

print("--- 8. Testing v0.5 Numerical Core (Tensors) ---")

// Test Tensor Creation
print("Testing Tensor Creation:")
t1 = tensor([[1, 2], [3, 4.5]])
t2 = tensor([[10, 20], [30, 40]])
print("Tensor t1:")
print(t1)
print("Tensor t2:")
print(t2)
print("")

// Test Element-wise Operations
print("Testing Element-wise Operations:")
print("t1 + t2:")
print(t1 + t2)
print("t2 - t1:")
print(t2 - t1)
print("")

// Test Scalar Broadcasting
print("Testing Scalar Broadcasting:")
print("t1 * 10:")
print(t1 * 10)
print("10 * t1:")
print(10 * t1)
print("")

// Test Matrix Multiplication
print("Testing Matrix Multiplication:")
t_matmul_res = matmul(t1, t2)
print("matmul(t1, t2):")
print(t_matmul_res)
print("")

// Test Tensor Indexing
print("Testing Tensor Indexing:")
print("Accessing t1[0, 1]:")
print(t1[0, 1]) 
print("Accessing row t1[1]:")
print(t1[1]) 
print("")

// Add new, more complex tests for slicing with steps
print("Testing Advanced Tensor Slicing with Steps:")
t_slice = tensor([[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]])
print("Original 4x4 Tensor:")
print(t_slice)

print("Every second row (t_slice[0:4:2, :]):")
print(t_slice[0:4:2, :])

print("Every second column from the first two rows (t_slice[0:2, 0:4:2]):")
print(t_slice[0:2, 0:4:2])

print("Reversed rows (t_slice[::-1, :]):")
print(t_slice[::-1, :])
print("")


// --- 9. Demonstrating v0.5 Tensor Runtime Errors ---
print("--- 9. Demonstrating v0.5 Tensor Runtime Errors ---")
print("Uncomment lines below one by one to test tensor errors.")

// Incompatible shapes for matrix multiplication
// matmul(t1, tensor([[1], [2], [3]]))
// Expected -> Runtime Error: Tensor shapes are incompatible for matrix multiplication.

// Index out of bounds
// print(t1[5, 5])
// Expected -> Runtime Error: Tensor index out of range.

print("")
print("--- v0.5 Test Suite Complete ---")