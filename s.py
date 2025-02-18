import os

file_path = "Linear_Regression_Model.pkl"  # Update if the file is in another directory

if os.path.exists(file_path):
    print("File exists:", file_path)
else:
    print("File not found:", os.getcwd())  # Check current directory
