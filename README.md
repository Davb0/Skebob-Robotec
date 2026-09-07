# Skebob-Defense

Skebob is a robot that follows things with a camera. Two cameras on a pan tilt turret
and two motors for driving and a laptop app you use to control it. You click on
something in the video and after that the robot tries to keep it in the middle of the
screen. The turret follows it by itself and if the target gets too far to the side then
the whole robot turns so the turret can catch up.

The first prototype was built for a freestyle robotics competition. That one was pretty
rushed and made out of whatever was lying around but it worked well enough to show the
idea and after the comp it just kept getting worked on instead of taken apart.

Where this is actually going: the goal is a mini SPAA basically a little anti air
system. Right now it just tracks and drives but once the tracking is good and the
driving is fully autonomous the plan is a shooting mechanism on the turret that shoots
airsoft balls at stuff in the air. So it finds a drone or whatever is flying and
keeps the turret on it and shoots it. Tracking and driving are step 1 and 2 and the
shooting is step 3.

## how it works right now

- FastAPI server on the robot and a PyQt6 app on the laptop and they talk over a
  websocket
- `/cam0` and `/cam1` are the video streams. cam0 is the RGB one with the tracking boxes
  drawn on it and cam1 is the IR one
- `/ws` sends the status like 10 times a second (turret angles, error in pixels, tracker
  state, motor speeds, closest obstacle) and the lidar scan 5 times a second
- the app sends back your clicks and turret nudges and drive commands and resets
- the app shows both cameras and a radar for the lidar and the status and a drive pad.
  you can drive with WASD or an xbox controller (needs pygame)

## the tracking

this was the hard part

- opencv KCF tracker with MOG2 background subtraction behind it
- KCF loses the target all the time especially if something walks in front of it so it
  also saves the HSV color histogram of what you clicked and uses that to find it again
- there's checks for if the box jumps too far in one frame or gets way bigger all of a
  sudden because that usually means it grabbed the wrong thing
- 3 states: idle locked and searching. when its searching the turret keeps going the
  last way it saw the target instead of just stopping

## the turret and the driving

- one PID for pan and one for tilt and they run off how far the target is from the
  center in pixels
- the error gets smoothed with an EMA and the integral is clamped and there's a deadzone
  and a max step per frame. the first version shook so hard it looked broken
- if the pan servo goes more than 30 degrees off center the motors spin the robot to
  bring it back. this used to go left right left right forever until hysteresis
  and a timeout got added
- the lidar gives a 360 scan for the radar and tells you how far the closest thing in
  front is

all the numbers are in `backend/config.py` (PID gains tracker thresholds camera indexes
frame size chassis angles) so you dont have to dig through the code

## running it

```bash
pip install -r requirements.txt
python backend/main.py     # http://localhost:8000
python app/main.py         # set ROBOT_HOST if its on another pc
```

everything runs on a laptop for now. the motors and servos are fake and just keep track
of what the numbers would be and the lidar is simulated and the cameras go to a test
pattern if nothing is plugged in. that way you can work on the code without the robot.

## next

- put it on the Jetson Orin Nano. Jetson.GPIO for the motors and PCA9685 over I2C for
  the servos and gstreamer for the IMX678 camera
- real RPLIDAR instead of the simulated one (`LIDAR_SIMULATED = False`)
- autonomous driving because right now you still drive it yourself
- the airsoft launcher on the turret and figuring out how to aim at stuff in the air

the files still in the root folder (tracker.py camera.py chassis.py motor.py arduino.ino
templates/) are the old raspberry pi + arduino version that had a website instead of an
app. they're only there for reference. dont run main.py from the root it crashes
