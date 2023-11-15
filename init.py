import subprocess

#all in one, dont need to run each file seperately
subprocess.call(["python", "model.py"])
subprocess.call(["python", "visualizations.py"])
subprocess.call(["python", "app.py"])