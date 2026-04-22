// // --- L298N Motor A (Left Side) ---
// const int enA = 9;  // Must be a PWM pin (~)
// const int in1 = 8;
// const int in2 = 7;

// // --- L298N Motor B (Right Side) ---
// const int enB = 3;  // Must be a PWM pin (~)
// const int in3 = 5;
// const int in4 = 4;

// void setup() {
//   // Start serial communication with Raspberry Pi
//   Serial.begin(9600);

//   // Set all motor control pins to outputs
//   pinMode(enA, OUTPUT);
//   pinMode(in1, OUTPUT);
//   pinMode(in2, OUTPUT);

//   pinMode(enB, OUTPUT);
//   pinMode(in3, OUTPUT);
//   pinMode(in4, OUTPUT);

//   // Make sure motors are off at startup
//   stopMotors();
// }

// void loop() {
//   // Check if data is available from Raspberry Pi
//   if (Serial.available() > 0) {
//     // Read the string until the newline character
//     String data = Serial.readStringUntil('\n');

//     // Find the comma to split left and right speeds
//     int commaIndex = data.indexOf(',');

//     if (commaIndex > 0) {
//       // Parse the integer values
//       int leftSpeed = data.substring(0, commaIndex).toInt();
//       int rightSpeed = data.substring(commaIndex + 1).toInt();

//       driveLeftMotor(leftSpeed);
//       driveRightMotor(rightSpeed);
//     }
//   }
// }

// void driveLeftMotor(int speed) {
//   if (speed > 0) {
//     // Forward
//     digitalWrite(in1, HIGH);
//     digitalWrite(in2, LOW);
//   } else if (speed < 0) {
//     // Reverse
//     digitalWrite(in1, LOW);
//     digitalWrite(in2, HIGH);
//   } else {
//     // Stop
//     digitalWrite(in1, LOW);
//     digitalWrite(in2, LOW);
//   }
//   // Send the absolute value (0-255) to the PWM pin
//   analogWrite(enA, abs(speed));
// }

// void driveRightMotor(int speed) {
//   if (speed > 0) {
//     // Forward
//     digitalWrite(in3, HIGH);
//     digitalWrite(in4, LOW);
//   } else if (speed < 0) {
//     // Reverse
//     digitalWrite(in3, LOW);
//     digitalWrite(in4, HIGH);
//   } else {
//     // Stop
//     digitalWrite(in3, LOW);
//     digitalWrite(in4, LOW);
//   }
//   // Send the absolute value (0-255) to the PWM pin
//   analogWrite(enB, abs(speed));
// }

// void stopMotors() {
//   digitalWrite(in1, LOW);
//   digitalWrite(in2, LOW);
//   analogWrite(enA, 0);

//   digitalWrite(in3, LOW);
//   digitalWrite(in4, LOW);
//   analogWrite(enB, 0);
// }
