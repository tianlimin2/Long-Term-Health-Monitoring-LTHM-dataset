import subprocess

# List of Python files to run in sequence
files_to_run = ['response_reflection.py', 'rouge_evaluation.py']

for file in files_to_run:
    print(f"Running {file}...")
    result = subprocess.run(['python', file], capture_output=True, text=True)
    
    # Check if the script ran successfully
    if result.returncode == 0:
        print(f"{file} ran successfully.")
    else:
        print(f"Error running {file}.")
        print(f"Error details: {result.stderr}")
