import cv2
import time
import RPi.GPIO as GPIO
import mediapipe as mp


# ============================================================
# GPIO CONFIGURATION
# ============================================================

# Left motor
ENA = 18
IN1 = 17
IN2 = 27

# Right motor
ENB = 13
IN3 = 22
IN4 = 25

MOTOR_SPEED = 70


# ============================================================
# GPIO SETUP
# ============================================================

GPIO.setmode(GPIO.BCM)

GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)

GPIO.setup(ENB, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)

left_pwm = GPIO.PWM(ENA, 1000)
right_pwm = GPIO.PWM(ENB, 1000)

left_pwm.start(0)
right_pwm.start(0)


# ============================================================
# MOTOR FUNCTIONS
# ============================================================

def stop_robot():
    left_pwm.ChangeDutyCycle(0)
    right_pwm.ChangeDutyCycle(0)

    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)

    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)


def forward():
    left_pwm.ChangeDutyCycle(MOTOR_SPEED)
    right_pwm.ChangeDutyCycle(MOTOR_SPEED)

    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)

    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)


def backward():
    left_pwm.ChangeDutyCycle(MOTOR_SPEED)
    right_pwm.ChangeDutyCycle(MOTOR_SPEED)

    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)

    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)


def turn_left():
    left_pwm.ChangeDutyCycle(MOTOR_SPEED)
    right_pwm.ChangeDutyCycle(MOTOR_SPEED)

    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)

    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)


def turn_right():
    left_pwm.ChangeDutyCycle(MOTOR_SPEED)
    right_pwm.ChangeDutyCycle(MOTOR_SPEED)

    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)

    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ============================================================
# COUNT FINGERS
# ============================================================

def count_fingers(hand_landmarks):

    landmarks = hand_landmarks.landmark

    fingers = 0

    # Thumb
    if landmarks[4].x < landmarks[3].x:
        fingers += 1

    # Index finger
    if landmarks[8].y < landmarks[6].y:
        fingers += 1

    # Middle finger
    if landmarks[12].y < landmarks[10].y:
        fingers += 1

    # Ring finger
    if landmarks[16].y < landmarks[14].y:
        fingers += 1

    # Little finger
    if landmarks[20].y < landmarks[18].y:
        fingers += 1

    return fingers


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open camera.")
    GPIO.cleanup()
    raise SystemExit


print("--------------------------------")
print("Hand Gesture Robot")
print("--------------------------------")
print("Show your hand to the camera.")
print("Press Q to quit.")


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        ret, frame = camera.read()

        if not ret:
            print("Camera frame unavailable.")
            break

        # Mirror camera image
        frame = cv2.flip(frame, 1)

        # Convert BGR -> RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb_frame)

        gesture = "NO HAND"

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            finger_count = count_fingers(
                hand_landmarks
            )

            # ----------------------------------------
            # Gesture mapping
            # ----------------------------------------

            if finger_count == 0:
                gesture = "BACKWARD"
                backward()

            elif finger_count == 1:
                gesture = "FORWARD"
                forward()

            elif finger_count == 2:
                gesture = "LEFT"
                turn_left()

            elif finger_count == 3:
                gesture = "RIGHT"
                turn_right()

            else:
                gesture = "STOP"
                stop_robot()

        else:
            # No hand = safe stop
            gesture = "NO HAND"
            stop_robot()


        # Display gesture
        cv2.putText(
            frame,
            f"Gesture: {gesture}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Q = Quit",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Hand Gesture Robot",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


except KeyboardInterrupt:

    print("Stopping robot...")


finally:

    stop_robot()

    left_pwm.stop()
    right_pwm.stop()

    camera.release()
    cv2.destroyAllWindows()

    GPIO.cleanup()

    print("Robot stopped.")
