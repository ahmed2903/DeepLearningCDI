import subprocess
import os

# List of folder paths where runDMO.py is located
path = '/scratch/ahmm1g15/Thesis/IDEs/'

folders = [ 'scan_-0.1V',
            'scan_0.1V',
            'scan_-0.2V',
            'scan_0.2V',  
            'scan_-0.3V',
            'scan_0.3V',  
            'scan_-1.5V',  
            'scan_1.5V',  
            'scan_-2.0V',  
            'scan_2.0V',  
            'scan_-3.0V',  
            'scan_-5.0V',  
            'scan_5.0V'
]

# Save the current directory to return back to it later
original_directory = os.getcwd()

try:
    # Loop through each folder
    for folder in folders:
        # Change the working directory to the current folder
        os.chdir(path+folder)
        
        # Run the script runDMO.py located in this folder 3 times
        for _ in range(3):
            # Execute the script using Python
            # Ensure that 'python' command is accessible or provide the full path to the python executable
            subprocess.run(["python", "runDyMO.py"])
            
        # Optionally, add a print statement to confirm which folder's script is being executed
        print(f"Finished executing runDMO.py 3 times in {folder}")

finally:
    # Ensure that we change back to the original directory
    os.chdir(original_directory)

print("All scripts executed.")