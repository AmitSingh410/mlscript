// ##########################################
// ## mlscript v0.3 - Comprehensive Test Suite
// ##########################################

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
sum = a + 5
product = b * 2
greeting = c + " mlscript!"

print("Sum (10 + 5):")
print(sum)
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

// --- 5. Demonstrating Improved Runtime Errors ---
print("--- 5. Demonstrating Improved Runtime Errors ---")
print("The following lines are commented out to prevent crashing the script.")
print("Uncomment them one by one to test the new error messages.")

// Uncomment the line below to test undefined variable error
// print(undefined_variable)
// Expected -> Runtime Error on line XXX: Undefined variable: undefined_variable

// Uncomment the line below to test undefined function error
// some_function_that_does_not_exist()
// Expected -> Runtime Error on line YYY: Undefined function 'some_function_that_does_not_exist'

print("--- Test Suite Complete ---")