

# **mlscript v0.2 Official Documentation**

## **Introduction**

This document provides the official documentation for mlscript version 0.2. It offers a comprehensive guide for developers, covering the language's core philosophy, setup instructions, language features, and practical examples.

### **Vision and Philosophy**

mlscript is a new programming language designed from the ground up to provide an intuitive and efficient environment for modern AI, Machine Learning, and Data Science workloads. The core vision is to bridge the gap between high-level, readable syntax and high-performance, low-level execution.

While many languages are general-purpose, mlscript is specialized. Its development is guided by the needs of data scientists, researchers, and machine learning engineers. The current version, v0.2, establishes the foundational syntax and architecture. Future versions will build upon this base to introduce first-class language constructs for core data science concepts, such as tensors, dataframes, and model layers, aiming to make complex numerical computation as natural as basic arithmetic.

### **The Hybrid Architecture: Python's Usability, C++'s Speed**

mlscript employs a powerful hybrid architecture to achieve its goals of simplicity and performance. It is not a standalone, self-contained executable; rather, it is implemented as a high-performance Python extension module. This design choice is fundamental to its operation and value.

The core of the language's interpreter is written in C++ and compiled to a native library, as defined by the project's build configuration.1 This C++ backend is responsible for parsing and executing

mlscript code, leveraging the speed of compiled C++. The connection between the Python host environment and the C++ backend is managed by pybind11, a library designed for creating seamless interoperability between the two languages.2

This architecture offers a significant advantage by combining the strengths of both ecosystems. Developers can leverage Python's vast and mature libraries for tasks like data loading, preprocessing, and visualization, while offloading computationally intensive algorithms to mlscript for execution at near-native C++ speeds. This model provides a compelling alternative to other performance-oriented Python solutions, offering the potential for a more tightly integrated and domain-specific syntax in the future.

## **Getting Started**

This section provides detailed instructions for setting up the build environment, compiling mlscript from source, and running your first program.

### **Prerequisites**

Before you can build mlscript, you must ensure your development environment has the necessary tools installed. The project relies on a modern C++ compiler, CMake for build automation, and a compatible Python version to host the module.

**Table 1: Prerequisites for mlscript v0.2**

| Tool | Required Version | Notes |
| :---- | :---- | :---- |
| C++ Compiler | C++17 Support | e.g., MSVC v142+ (VS 2019+), GCC 7+, Apple Clang 10+ |
| CMake | 3.15+ | The core build system generator for the project.1 |
| Python | 3.8+ | The host environment for the mlscript module.2 |
| pip | Included w/ Python | Used to install the pybind11 dependency.3 |

### **Environment Setup and Build Instructions**

The following instructions provide a recommended setup path for Windows, macOS, and Linux.

#### **On Windows**

1. **Install C++ Toolchain:** Download and install Visual Studio 2019 or a newer version from Microsoft. During installation, select the **Desktop development with C++** workload. This will install the required MSVC compiler with C++17 support.4  
2. **Install Python:** Download the latest Python 3 installer from the official website, python.org.5 During installation, it is crucial to check the box labeled  
   **Add Python to PATH**.  
3. **Install CMake:** Download and run the official CMake installer from cmake.org.6 Ensure that you select the option to add CMake to the system PATH for all users or the current user.  
4. **Install pybind11:** Open a Command Prompt or PowerShell and run the following command to install the required Python package 7:  
   Shell  
   py \-m pip install pybind11

#### **On macOS**

1. **Install C++ Toolchain:** Open the Terminal application and run the following command. This will prompt you to install the Xcode Command Line Tools, which include the Apple Clang C++ compiler.8  
   Shell  
   xcode-select \--install

2. **Install Homebrew:** If you do not have the Homebrew package manager, install it by following the instructions on its official website, brew.sh.  
3. **Install Python & CMake:** Use Homebrew to install the latest versions of Python and CMake.9  
   Shell  
   brew install python cmake

4. **Install pybind11:** Use pip to install the pybind11 package 11:  
   Shell  
   python3 \-m pip install pybind11

#### **On Linux (Debian/Ubuntu)**

1. **Install C++ Toolchain & Dependencies:** Open a terminal and use the apt package manager to install the essential build tools, including the GCC C++ compiler and GDB debugger.12  
   Shell  
   sudo apt-get update  
   sudo apt-get install build-essential gdb

2. **Install Python & CMake:** Install the Python 3 development headers, pip, and CMake.13  
   Shell  
   sudo apt-get install python3-dev python3-pip cmake

3. **Install pybind11:** Use pip to install the pybind11 package:  
   Shell  
   python3 \-m pip install pybind11

#### **Compiling the Module**

Once all prerequisites are installed, you can compile the mlscript module using CMake.

1. **Clone the Repository:** First, clone the mlscript source code repository to your local machine.  
   Shell  
   git clone https://github.com/example/mlscript.git  
   cd mlscript

2. **Configure with CMake:** Create a build directory and run CMake from within it. This command inspects your system, finds the required tools, and generates the native build files (e.g., a Visual Studio solution on Windows or a Makefile on Linux/macOS).14  
   Shell
   cd cpp_backend  
   mkdir build  
   cd build  
   cmake..

3. **Build with CMake:** Execute the build command. This command invokes the native build tool (like MSVC or Make) to compile the C++ source code and link it into a Python extension module.14 The  
   \--config Release flag ensures an optimized build.  
   Shell  
   cmake \--build. \--config Release

4. **Copy the Module:** After the build completes, you must manually copy the compiled module to the project's root directory (the same directory that contains main.py).  
   * **On Windows:** The file will be named mlscript.cp313-win_amd64.pyd and is typically located in the build\\Release directory.  
   * **On macOS/Linux:** The file will be named with a .so extension and is typically located directly in the build directory.

### **Running mlscript**

Because mlscript is a Python module, you interact with it via the main.py Python host script.

#### **The Interactive REPL (Read-Eval-Print Loop)**

The REPL is ideal for quick experiments and learning the language syntax. To start it, run main.py from the project's root directory without any arguments.

Shell

python main.py

You will be greeted by the mlscript prompt, where you can enter code one line at a time.

**Example REPL Session:**

mlscript v0.2 REPL  
\>\>\> message \= "Hello, mlscript\!"  
\>\>\> print(message)  
Hello, mlscript\!  
\>\>\> x \= 10 \* 5  
\>\>\> print(x)  
50  
\>\>\>

#### **Executing Script Files**

For larger programs, you can save your code in a file with a .ms extension and execute it using the main.py script.

**Example hello.ms file:**

// A simple mlscript program  
name \= "World"  
print("Hello, " \+ name \+ "\!")

To run this script, pass its path to main.py:

Shell

python main.py hello.ms

**Output:**

Hello, World\!

## **Language Guide**

This section serves as a comprehensive reference for all language features available in mlscript v0.2.

### **Comments**

Comments are used to annotate code and are ignored by the interpreter. mlscript supports single-line comments, which start with // and extend to the end of the line.

**Syntax:**

// This is a comment.

**Example:**

// This line is ignored.  
x \= 10 // This part of the line is also ignored.  
print(x) // Prints 10

### **Data Types**

mlscript v0.2 supports three primitive data types.

* **Integer:** A 64-bit signed integer for whole numbers (e.g., 10, \-5, 0).  
* **Float:** A 64-bit double-precision floating-point number for values with a fractional component (e.g., 3.14, \-0.001, 10.0).  
* **String:** A sequence of characters enclosed in double quotes ("). Strings are UTF-8 encoded.

**Example:**

my\_integer \= 42  
my\_float \= 2.718  
my\_string \= "mlscript"

print(my\_integer)  
print(my\_float)  
print(my\_string)

### **Variables and Scope**

Variables are used to store data. They are defined within a specific scope and are created on first assignment.

#### **Variable Assignment**

Variables are created when they are first assigned a value. mlscript uses a Python-like approach where you do not need to declare variables before using them. Simply assign a value to a name, and the variable is created.

**Syntax:**

\<identifier\> \= \<expression\>

A variable's value can be updated by assigning a new value to it.

**Example:**

// Initial assignment creates the variable  
score \= 100  
print(score) // Output: 100

// Re-assignment  
score \= 150  
print(score) // Output: 150

#### **Scope**

mlscript uses lexical (or static) scoping, which means a variable's visibility is determined by its location within the source code. A scope is defined by a block of code enclosed in curly braces ({...}), such as a function body or the body of a control flow statement.

A variable declared inside a block is local to that block and is not accessible from the outside. Inner scopes can access variables from their containing outer scopes.

**Example:**

global\_var \= "I am global"

{  
  inner\_var \= "I am local"  
  print(global\_var) // Accessible: prints "I am global"  
  print(inner\_var)  // Accessible: prints "I am local"  
}

print(global\_var) // Accessible: prints "I am global"  
// print(inner\_var) // This would cause an error, inner\_var is not defined here.

### **Operators**

mlscript supports standard arithmetic and comparison operators.

#### **Arithmetic Operators**

These operators perform mathematical calculations on numerical types.

**Table 2: Arithmetic Operators**

| Operator | Description | Example |
| :---- | :---- | :---- |
| \+ | Addition (or String concatenation) | 5 \+ 2 |
| \- | Subtraction | 5 \- 2 |
| \* | Multiplication | 5 \* 2 |
| / | Division | 5.0 / 2.0 |

**Example:**

a \= 10  
b \= 4  
print(a \+ b) // 14  
print(a \- b) // 6  
print(a \* b) // 40  
print(a / b) // 2.5

greeting \= "Hello"  
subject \= "World"  
print(greeting \+ ", " \+ subject \+ "\!") // "Hello, World\!"

#### **Comparison Operators**

These operators compare two values and evaluate to a boolean result, which is internally represented and used by control flow statements like if and while.

**Table 3: Comparison Operators**

| Operator | Description | Example |
| :---- | :---- | :---- |
| \== | Equal to | x \== 5 |
| \!= | Not equal to | x\!= 5 |
| \< | Less than | x \< 5 |
| \<= | Less than or equal to | x \<= 5 |
| \> | Greater than | x \> 5 |
| \>= | Greater than or equal to | x \>= 5 |

**Example:**

x \= 10  
y \= 20

if (x \< y) {  
  print("x is less than y")  
}

if (x \* 2 \== y) {  
  print("y is double of x")  
}

### **Control Flow**

Control flow statements allow you to direct the execution of your program based on certain conditions or to repeat blocks of code.

#### **Conditional Statements: if/elif/else**

These statements execute different blocks of code based on a condition. The block of code must be enclosed in curly braces ({...}).

**Syntax:**

if (\<condition1\>) {  
  // Block executed if condition1 is true  
} elif (\<condition2\>) {  
  // Block executed if condition1 is false and condition2 is true  
} else {  
  // Block executed if all preceding conditions are false  
}

**Example:**

grade \= 85

if (grade \>= 90\) {  
  print("Grade: A")  
} elif (grade \>= 80\) {  
  print("Grade: B")  
} else {  
  print("Grade: C or lower")  
}  
// Output: Grade: B

#### **while Loops**

A while loop repeatedly executes a block of code, enclosed in curly braces ({...}), as long as a given condition remains true.

**Syntax:**

while (\<condition\>) {  
  // This block repeats as long as condition is true  
}

**Example:**

countdown \= 3  
while (countdown \> 0\) {  
  print(countdown)  
  countdown \= countdown \- 1  
}  
print("Liftoff\!")  
// Output:  
// 3  
// 2  
// 1  
// Liftoff\!

#### **for Loops**

A for loop is used for iterating over a sequence. In v0.2, the only available sequence generator is range(\<start\>, \<end\>), which creates a sequence of integers starting from \<start\> and ending just before \<end\>. The loop body must be enclosed in curly braces ({...}).

**Syntax:**

for \<variable\> in range(\<start\>, \<end\>) {  
  // This block executes for each number in the range  
}

**Example:**

// Print numbers from 1 up to (but not including) 5  
for i in range(1, 5\) {  
  print("Current number: " \+ i)  
}  
// Output:  
// Current number: 1  
// Current number: 2  
// Current number: 3  
// Current number: 4

### **Functions**

Functions are reusable blocks of code that perform a specific task. They help organize code into logical, modular units. The function body must be enclosed in curly braces ({...}).

#### **Definition and Calling**

Functions are defined using the fun keyword. They can accept zero or more parameters.

**Syntax:**

fun \<function\_name\>(\<param1\>, \<param2\>,...) {  
  // Function body  
}

**Example:**

// Define a function to greet a user  
fun greet(name) {  
  print("Hello, " \+ name \+ "\!")  
}

// Call the function  
greet("Alice")   // Output: Hello, Alice\!  
greet("Bob")     // Output: Hello, Bob\!

#### **The return Statement**

The return statement exits a function and can optionally pass a value back to the caller. If return is not used, the function implicitly returns a null-like value.

**Example:**

fun add(a, b) {  
  return a \+ b  
}

sum \= add(15, 27\)  
print("The sum is: " \+ sum) // Output: The sum is: 42

## **Full Showcase: Calculating Basic Data Statistics**

This final example demonstrates how the features of mlscript v0.2 can work together to solve a practical problem. The following script calculates the count, sum, average, minimum, and maximum for a simulated list of data points. This task is a foundational element of data analysis and showcases the language's potential for more complex numerical tasks.

The script is structured with functions for clarity and reusability, uses loops for iteration, conditionals for logic, and variables to store state.

// mlscript v0.2 Showcase: Basic Data Statistics

// A helper function to neatly print the final results.  
// This demonstrates function definition and parameter passing.  
fun print\_stats(count, sum, avg, min\_val, max\_val) {  
  print("--- Data Statistics \---")  
  print("Count: " \+ count)  
  print("Sum: " \+ sum)  
  print("Average: " \+ avg)  
  print("Minimum: " \+ min\_val)  
  print("Maximum: " \+ max\_val)  
  print("-----------------------")  
}

// The main function that performs all calculations.  
// This encapsulates the core logic of the program.  
fun calculate\_stats() {  
  // In a future version, this data might come from a file or a list data type.  
  // For v0.2, we simulate a list with individual variables.  
  data0 \= 85.5  
  data1 \= 92.0  
  data2 \= 78.2  
  data3 \= 61.0  
  data4 \= 95.8  
  data5 \= 88.5

  // Initialize variables to store our calculated statistics.  
  count \= 0  
  sum \= 0.0  
  min\_val \= 1000.0 // Start with a very high number  
  max\_val \= \-1000.0 // Start with a very low number

  // The 'for' loop iterates from 0 to 5 to process our 6 data points.  
  // This is the core of our data processing logic.  
  for i in range(0, 6\) {  
    current\_val \= 0.0

    // Use if/elif to select the correct data point for the current iteration.  
    // This simulates accessing an element from a list by its index.  
    if (i \== 0\) { current\_val \= data0 }  
    elif (i \== 1\) { current\_val \= data1 }  
    elif (i \== 2\) { current\_val \= data2 }  
    elif (i \== 3\) { current\_val \= data3 }  
    elif (i \== 4\) { current\_val \= data4 }  
    elif (i \== 5\) { current\_val \= data5 }

    // Update the sum and count on each iteration.  
    sum \= sum \+ current\_val  
    count \= count \+ 1

    // Use conditional logic to find the minimum value.  
    if (current\_val \< min\_val) {  
      min\_val \= current\_val  
    }

    // Use conditional logic to find the maximum value.  
    if (current\_val \> max\_val) {  
      max\_val \= current\_val  
    }  
  }

  // After the loop, calculate the average.  
  average \= 0.0  
  if (count \> 0\) {  
    average \= sum / count  
  }

  // Call our helper function to display the results.  
  print\_stats(count, sum, average, min\_val, max\_val)  
}

// This is the entry point of our script.  
// The program execution begins here.  
print("Starting statistical analysis...")  
calculate\_stats()  
print("Analysis complete.")
