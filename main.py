# main loop
import subprocess

print("mango tuff")

# va da crash btw.

chassis = subprocess.run(["python", "chassis.py"], capture_output=True, text=True)
camera = subprocess.run(["python", "camera.py"], capture_output=True, text=True)
print("Camera:", camera.stdout)
print("Chassis:", chassis.stdout)
