// ######################################################
// ## mlscript - Comprehensive Test Suite
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

print("--- 7. Testing v0.5 Numerical Core (Tensors) ---")

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


// ######################################################
// ## v0.6 - NEW FEATURE TESTS
// ######################################################

print("--- 8. Testing v0.6 Python Interoperability ---")

// Test 1: Importing a Python module
print("Testing 'import' statement...")
import "numpy" as np
print("Successfully imported numpy as 'np'.")
print(np)
print("")

// Test 2: Creating a mlscript Tensor from a NumPy array
print("Testing NumPy -> mlscript bridge...")
np_array = np.array([[10, 20], [30, 40]])
print("Created NumPy array using np.array():")
print(np_array)
t_from_np = tensor(np_array)
print("Created mlscript Tensor from NumPy array:")
print(t_from_np)
print("")

// Test 3: Creating a NumPy array from an mlscript Tensor
print("Testing mlscript -> NumPy bridge...")
t_to_np = tensor([[1.5, 2.5], [3.5, 4.5]])
print("Original mlscript Tensor:")
print(t_to_np)
np_from_t = np.array(t_to_np * 10)
print("Created NumPy array from (t_to_np * 10):")
print(np_from_t)
print("")

// Test 4: Chained operations
print("Testing chained operations...")
chained_tensor = tensor(np.array(t_from_np) + 100)
print("Result of 'tensor(np.array(t_from_np) + 100)':")
print(chained_tensor)
print("")

//Test 5: Automatic Type Conversion
print("Testing automatic type conversion (mlscript list -> C++ vector)...")
ml_list = [[100.0, 200.0], [300.0, 400.0]]
t_from_ml_list = tensor(ml_list)
print("Tensor created from native mlscript list:")
print(t_from_ml_list)
print("")


// ######################################################
// ## v0.7 - NEW FEATURE TESTS
// ######################################################

print("")
print("--- 9. Testing v0.7 Automatic Differentiation ---")

// Test 1: Simple addition
print("Test 1: Gradient of Addition")
a = tensor([[1, 1], [1, 1]])
b = tensor([[2, 2], [2, 2]])
c = a + b
loss_add = c.sum()
loss_add.backward()

print("a = [[1,1],[1,1]]")
print("b = [[2,2],[2,2]]")
print("loss = (a + b).sum()")
print("a.grad (should be all 1s):")
print(a.grad)
print("b.grad (should be all 1s):")
print(b.grad)
print("")


// Test 2: Element-wise multiplication
print("Test 2: Gradient of Multiplication")
d = tensor([[2, 3], [4, 5]])
e = tensor([[10, 10], [10, 10]])
f = d * e
loss_mul = f.sum()
loss_mul.backward()

print("d = [[2,3],[4,5]]")
print("e = [[10,10],[10,10]]")
print("loss = (d * e).sum()")
print("d.grad (should be equal to e):")
print(d.grad)
print("e.grad (should be equal to d):")
print(e.grad)
print("")


// Test 3: Chained operations
print("Test 3: Chained Operations (y = (a*b)+c)")
chain_a = tensor([[2, 2]])
chain_b = tensor([[3, 3]])
chain_c = tensor([[4, 4]])

prod = chain_a * chain_b 
y = prod + chain_c 
loss_chain = y.sum()
loss_chain.backward()

print("a = [[2,2]], b = [[3,3]], c = [[4,4]]")
print("loss = ((a*b)+c).sum()")
print("a.grad (should be equal to b):")
print(chain_a.grad)
print("b.grad (should be equal to a):")
print(chain_b.grad)
print("c.grad (should be all 1s):")
print(chain_c.grad)
print("")


// Test 4: no_grad context
print("Test 4: Verifying 'no_grad' context")
ng_a = tensor([[10, 10]]) 
ng_b = tensor([[20, 20]]) 

with no_grad {
    ng_c = ng_a * ng_b 
    print("Inside no_grad, ng_c is computed:")
    print(ng_c)
}

final_output = ng_a * 3 
loss_no_grad = final_output.sum()
loss_no_grad.backward()

print("final_loss = (ng_a * 3).sum()")
print("ng_a.grad (should be 3s, from the final computation):")
print(ng_a.grad)
print("ng_b.grad (should be 0s, as it was used only in no_grad):")
print(ng_b.grad)
print("ng_c has no grad, as it was created in no_grad context.")
print("")

// ######################################################
// ## v0.8 - NEW FEATURE TESTS
// ######################################################

print("")
print("--- 10. Testing v0.8 Language Polish ---")

// Test Tuples
print("Testing Tuples:")
my_tuple = ("hello", 10, true)
single_tuple = (99,)
empty_tuple = ()
print(my_tuple)
print(single_tuple)
print(empty_tuple)
print("Iterating over my_tuple:")
for item in my_tuple {
    print(item)
}
print("")

// Test Default Function Arguments
print("Testing Default Arguments:")
fun greet(name, punctuation="!") {
    print("Greeting: " + name + punctuation)
}
greet("mlscript")
greet("Partner", "!!!")
print("")

// Test 'in' and 'not in' operators
print("Testing 'in' and 'not in':")
check_list = [1, "a", 2]
check_dict = {"key": "value"}
check_str = "mlscript"
check_tuple = (1, 2, 3)
print("1 in [1, 'a', 2]:")
print(1 in check_list)
print("3 not in [1, 'a', 2]:")
print(3 not in check_list)
print("'key' in {'key': 'value'}:")
print("key" in check_dict)
print("'scr' in 'mlscript':")
print("scr" in check_str)
print("3 in (1, 2, 3):")
print(3 in check_tuple)
print("")

print("--- 11. Testing v0.8 Advanced Control Flow ---")
// Test break and continue
print("Testing 'break' and 'continue':")
for i in range(10) {
    if (i == 3) {
        print("...continuing at 3...")
        continue
    }
    if (i == 7) {
        print("...breaking at 7...")
        break
    }
    print(i)
}
print("")

// Test try/catch/finally
print("Testing 'try/catch/finally':")
try {
    print("Inside try block (will throw).")
    throw "This is a test error!"
} catch (e) {
    print("Caught an error:")
    print(e)
} finally {
    print("Finally block executed (after error).")
}

try {
    print("Inside try block (will succeed).")
} catch (e) {
    print("This should not be printed.")
} finally {
    print("Finally block executed (after success).")
}
print("")

print("--- 12. Testing v0.8 OOP (Classes and Inheritance) ---")
class Pet {
    fun init(self, name) {
        self.name = name
    }
    fun speak(self) {
        return self.name + " makes a sound."
    }
}

class Dog inherits Pet {
    fun init(self, name, breed) {
        super.init(name)
        self.breed = breed
    }
    fun speak(self) {
        return self.name + " says woof!"
    }
}

class Mixin {
    fun get_id(self) {
        return 12345
    }
}

class RoboDog inherits Dog, Mixin {
    fun speak(self) {
        return "RoboDog override: " + super.speak()
    }
}

print("Testing Single Inheritance:")
my_dog = Dog("Rex", "German Shepherd")
print(my_dog.speak()) // Expected: Rex says woof!
print("")

print("Testing Multiple Inheritance and MRO:")
my_robo_dog = RoboDog("Fido", "Cyber-Terrier")
print(my_robo_dog.speak()) // Expected: RoboDog override: Fido says woof!
print(my_robo_dog.get_id()) // Expected: 12345
print("")


print("--- 13. Testing v0.8 Performance (OpenMP) ---")
print("The following is a benchmark and not a correctness test.")
print("If OpenMP is working, you should see high CPU usage during matmul.")
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

print("--- v0.8 Test Suite Complete ---")