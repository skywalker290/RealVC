import os

# Print the current working directory
print("Current working directory:", os.getcwd())

# Change to a new directory
new_directory = 'RVC'
os.chdir(new_directory)

# Print the new working directory
print("New working directory:", os.getcwd())
