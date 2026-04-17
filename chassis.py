# cod pt. chasis pt raspeberry pi/

import sys
import time

import pygame
import serial

# --- CONFIGURATION ---
# The serial port might be /dev/ttyACM0, /dev/ttyACM1, or /dev/ttyUSB0
# You can check by running `ls /dev/tty*` in your Pi's terminal before and after plugging in the Arduino.
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 9600

# Initialize Serial Connection
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Give Arduino a moment to reset after connection
    print(f"Connected to Arduino on {SERIAL_PORT}")
except serial.SerialException:
    print(f"Error: Could not open serial port {SERIAL_PORT}. Check your connection.")
    sys.exit(1)

# Initialize Pygame and Joystick
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("Error: No joystick detected. Please connect your Xbox controller.")
    sys.exit(1)

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Initialized Joystick: {joystick.get_name()}")

try:
    while True:
        pygame.event.pump()  # Update internal state of pygame

        # Xbox controllers on Linux typically use Axis 1 for Left Y and Axis 4 for Right Y.
        # Pygame axes go from -1.0 (up) to 1.0 (down). We multiply by -1 to make up positive.
        left_y = -joystick.get_axis(1)
        right_y = -joystick.get_axis(4)

        # Convert to PWM range (-255 to 255)
        left_speed = int(left_y * 255)
        right_speed = int(right_y * 255)

        # Apply a "deadzone" so the motors don't whine when the sticks are at rest
        if abs(left_speed) < 30:
            left_speed = 0
        if abs(right_speed) < 30:
            right_speed = 0

        # Format the command: "LeftSpeed,RightSpeed\n"
        command = f"{left_speed},{right_speed}\n"

        # Send to Arduino
        arduino.write(command.encode("utf-8"))

        # Run at ~20 updates per second to not flood the serial buffer
        time.sleep(0.05)

except KeyboardInterrupt:
    print("Stopping...")
    # Send a stop command before exiting
    arduino.write("0,0\n".encode("utf-8"))
    arduino.close()
    pygame.quit()
