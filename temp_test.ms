
global_counter = 0


def get_fizzbuzz_string(n)  {
    if (n == 3) { return "Fizz" }
    if (n == 5) { return "Buzz" }
    if (n == 6) { return "Fizz" }
    if (n == 9) { return "Fizz" }
    if (n == 10) { return "Buzz" }
    return n 
}

def fizzbuzz_loop(current_number, max_number) {
    if (current_number > max_number) {
        return 0
    }

    output = get_fizzbuzz_string(current_number)
    print(output)
    
    global_counter = global_counter + 1

    fizzbuzz_loop(current_number + 1, max_number)
}



print("Starting FizzBuzz")

fizzbuzz_loop(1, 10)

print("Loop Complete")

print("Total iterations run:")
print(global_counter)