// Global State 
tasks_completed = 0
system_status = "OK"

// Function Definitions 
fun process_item(item_id, quality_score) {
    print("--- Processing item: ---" )
    print(item_id)

    if (quality_score < 50) {
        system_status = "Warning: Low quality item detected."
        return 0
    } elif (quality_score == 100) {
        print("Perfect quality item!")
        tasks_completed = tasks_completed + 1
        return 1
    } else {
        tasks_completed = tasks_completed + 1
        return 1
    }
}

fun run_batch_job(job_id) {
    print("--- Starting Batch Job ---")
    print(job_id)
    
    for i in range(3) {
        //'score' is local to the for-loop's scope
        score = 50 + i * 20 
        process_item(i, score)
    }

    print("--- Batch Job Complete ---")
    return tasks_completed
}


// Main Execution Logic
print("--- System Initializing --- ")
run_batch_job("JOB-A1")

print("--- Verifying System Status ---")
print(system_status)

print("--- Running Cooldown Protocol ---")
cooldown = 3
while (cooldown > 0) {
    print("System cooling down...")
    cooldown = cooldown - 1
}

print("Final Report ")
print("Total tasks completed:")
print(tasks_completed)