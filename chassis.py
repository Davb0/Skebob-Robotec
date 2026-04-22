import sys
import time

import pygame
from gpiozero import Motor

# --- HARDWARE CONFIGURATION ---
# baga aici gpio corect sau ceva idfk.
try:
    left_motor = Motor(forward=5, backward=6, enable=12)
    right_motor = Motor(forward=23, backward=24, enable=13)
    print("Motors initialized successfully.")
except Exception as e:
    print(f"Error initializing GPIO: {e}")
    sys.exit(1)

# --- CONTROLLER SETUP ---
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("Error: No joystick detected. Please connect your Xbox controller.")
    sys.exit(1)

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Initialized Joystick: {joystick.get_name()}")


# --- HELPER FUNCTION ---
def drive_motor(motor, speed):
    """
    Drives a gpiozero Motor object.
    Speed should be a float between -1.0 (full reverse) and 1.0 (full forward).
    """
    if speed > 0:
        motor.forward(speed)
    elif speed < 0:
        motor.backward(abs(speed))
    else:
        motor.stop()


# --- MAIN LOOP ---
try:
    while True:
        pygame.event.pump()  # Update internal state of pygame

        # Xbox controllers typically use Axis 1 for Left Y and Axis 4 for Right Y.
        # Pygame axes go from -1.0 (up) to 1.0 (down).
        # We multiply by -1 so that pushing UP yields a positive number.
        left_y = -joystick.get_axis(1)
        right_y = -joystick.get_axis(4)

        # Apply a "deadzone" to prevent drift when sticks are at rest
        deadzone = 0.15

        if abs(left_y) < deadzone:
            left_y = 0.0
        if abs(right_y) < deadzone:
            right_y = 0.0

        # Drive the motors (gpiozero expects a float between 0.0 and 1.0 for speed)
        drive_motor(left_motor, left_y)
        drive_motor(right_motor, right_y)

        # Run at ~20 updates per second to save CPU
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopping robot...")
    # Ensure motors turn off when the script is killed
    left_motor.stop()
    right_motor.stop()
    pygame.quit()
    sys.exit(0)