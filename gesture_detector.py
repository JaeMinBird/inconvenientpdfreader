import cv2
import mediapipe as mp
import numpy as np
import time

class GestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Face mesh for detecting lips
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Gesture detection variables
        self.previous_x = None
        self.start_x = None
        self.swipe_threshold_left = 0.10  # Minimum distance for LEFT swipe (10% - more sensitive)
        self.swipe_threshold_right = 0.12  # Minimum distance for RIGHT swipe (12%)
        self.fast_swipe_threshold = 0.20  # Threshold for fast/large swipes (20% of screen width)
        self.gesture_cooldown = 1.5   # Cooldown time between gestures
        self.last_gesture_time = 0
        self.gesture_started = False
        self.movement_frames = 0
        self.min_frames = 3  # Minimum frames for a valid swipe
        self.max_frames = 25  # Maximum frames for a valid swipe (increased from 20)

        # Finger lick detection
        self.finger_licked = False
        self.lick_timeout = 5.0  # Finger lick is valid for 5 seconds (increased from 3)
        self.lick_time = 0
        self.was_at_lips = False  # Track if finger was just at lips

        # Track position history for better swipe detection
        self.position_history = []
        self.thumb_history = []
        self.history_size = 15  # Track last 15 positions
        
    def is_finger_lick_pose(self, hand_landmarks):
        """
        Detect if the hand is in a "finger lick" pose:
        - Index finger extended and pointing up
        - Near the top of the frame (near mouth level)
        - Other fingers curled down (relaxed requirements)
        """
        # Get landmark positions
        index_tip = hand_landmarks.landmark[8]  # Index finger tip
        index_pip = hand_landmarks.landmark[6]  # Index finger middle joint
        index_mcp = hand_landmarks.landmark[5]  # Index finger base
        middle_tip = hand_landmarks.landmark[12]  # Middle finger tip
        middle_pip = hand_landmarks.landmark[10]  # Middle finger middle joint
        ring_tip = hand_landmarks.landmark[16]  # Ring finger tip
        pinky_tip = hand_landmarks.landmark[20]  # Pinky tip
        wrist = hand_landmarks.landmark[0]

        # Check if index finger is extended (tip is above middle joint)
        index_extended = index_tip.y < index_pip.y - 0.05

        # Check if index finger is in upper portion of frame (near mouth) - more relaxed
        index_near_top = index_tip.y < 0.5  # Top 50% of frame (was 40%)

        # More relaxed: just check that index is the most extended finger
        # (other fingers should be lower than index tip)
        index_most_extended = (index_tip.y < middle_tip.y - 0.05 and
                              index_tip.y < ring_tip.y - 0.05 and
                              index_tip.y < pinky_tip.y - 0.05)

        # Accept if index is extended, near top, and is the highest finger
        is_lick_pose = index_extended and index_near_top and index_most_extended

        return is_lick_pose

    def detect_gesture(self, frame):
        """
        Detect hand swipe gestures in the frame
        Returns: 'left', 'right', or None
        Requires "finger lick" gesture first to enable page turning
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb_frame)
        face_results = self.face_mesh.process(rgb_frame)

        gesture = None
        current_time = time.time()

        # Check if finger lick has expired
        if self.finger_licked and (current_time - self.lick_time > self.lick_timeout):
            self.finger_licked = False
            print("[GESTURE] Finger lick expired")

        # Get lip position if face is detected
        lip_region = None
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]

            # Draw minimal face mesh (just lips area)
            # Lip landmarks: upper lip = 13, lower lip = 14
            # Get center of lips (landmark 13 for upper lip center)
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]

            # Calculate lip region center and create a detection zone
            lip_center_x = (upper_lip.x + lower_lip.x) / 2
            lip_center_y = (upper_lip.y + lower_lip.y) / 2

            # Define lip region radius (in normalized coordinates)
            lip_radius = 0.08  # 8% of frame size
            lip_region = {
                'x': lip_center_x,
                'y': lip_center_y,
                'radius': lip_radius
            }

            # Draw lip detection zone on frame
            h, w = frame.shape[:2]
            lip_x = int(lip_center_x * w)
            lip_y = int(lip_center_y * h)
            lip_r = int(lip_radius * min(w, h))
            cv2.circle(frame, (lip_x, lip_y), lip_r, (255, 200, 0), 2)
            cv2.putText(frame, "LIP ZONE", (lip_x - 40, lip_y - lip_r - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Check for finger touching lips (if face is detected)
                finger_at_lips = False
                if lip_region is not None:
                    # Check if index finger tip is near lips
                    index_tip = hand_landmarks.landmark[8]

                    # Calculate distance between finger and lips
                    dx = index_tip.x - lip_region['x']
                    dy = index_tip.y - lip_region['y']
                    distance = np.sqrt(dx**2 + dy**2)

                    # Check if finger is within lip region
                    if distance < lip_region['radius']:
                        finger_at_lips = True
                        # Draw connection line
                        h, w = frame.shape[:2]
                        cv2.line(frame,
                                (int(index_tip.x * w), int(index_tip.y * h)),
                                (int(lip_region['x'] * w), int(lip_region['y'] * h)),
                                (0, 255, 255), 2)

                # Check for finger lick pose OR finger touching lips
                is_lick_pose = self.is_finger_lick_pose(hand_landmarks)
                currently_at_lips = is_lick_pose or finger_at_lips

                if currently_at_lips:
                    self.was_at_lips = True
                    if not self.finger_licked:
                        self.finger_licked = True
                        self.lick_time = current_time
                        # Reset cooldown when finger is licked - you're ready for a new turn
                        self.last_gesture_time = 0
                        lick_method = "finger touched lips" if finger_at_lips else "finger lick pose"
                        print(f"[GESTURE] Licked! ({lick_method}) Ready to turn page (cooldown reset)")
                    cv2.putText(frame, "READY TO TURN PAGE", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                elif self.was_at_lips and self.finger_licked:
                    # Just left lips - show transition state
                    self.was_at_lips = False
                    time_remaining = self.lick_timeout - (current_time - self.lick_time)
                    cv2.putText(frame, f"SWIPE NOW! ({time_remaining:.1f}s)", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Get middle finger tip and thumb tip positions for swipe tracking
                middle_finger_tip = hand_landmarks.landmark[12]
                thumb_tip = hand_landmarks.landmark[4]
                current_x = middle_finger_tip.x
                thumb_x = thumb_tip.x

                # Only track swipes if finger has been licked and NOT currently at lips
                if self.finger_licked and not currently_at_lips:
                    # Add current position to history
                    self.position_history.append(current_x)
                    self.thumb_history.append(thumb_x)
                    if len(self.position_history) > self.history_size:
                        self.position_history.pop(0)
                    if len(self.thumb_history) > self.history_size:
                        self.thumb_history.pop(0)

                    # Start tracking a new gesture
                    if not self.gesture_started:
                        self.start_x = current_x
                        self.gesture_started = True
                        self.movement_frames = 0
                    else:
                        self.movement_frames += 1

                        # Check if we're in cooldown period - ignore all gestures
                        if current_time - self.last_gesture_time <= self.gesture_cooldown:
                            # Still in cooldown, reset tracking
                            self.gesture_started = False
                            self.position_history.clear()
                            self.thumb_history.clear()
                            cv2.putText(frame, "COOLDOWN", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        # Check if enough frames have passed for a valid swipe
                        elif self.movement_frames >= self.min_frames and len(self.position_history) >= 5 and len(self.thumb_history) >= 5:
                            # Use the first and last positions from history for more stable detection
                            # This ignores small counter-movements
                            start_pos = self.position_history[0]
                            end_pos = self.position_history[-1]
                            total_movement = end_pos - start_pos

                            # Track thumb movement as well
                            thumb_start = self.thumb_history[0]
                            thumb_end = self.thumb_history[-1]
                            thumb_movement = thumb_end - thumb_start

                            # Calculate average speed (movement per frame)
                            avg_speed = abs(total_movement) / len(self.position_history)

                            # Also check that the overall trend is consistent
                            # Count how many movements are in the same direction
                            movements_right = 0
                            movements_left = 0
                            thumb_right = 0
                            thumb_left = 0

                            for i in range(1, len(self.position_history)):
                                diff = self.position_history[i] - self.position_history[i-1]
                                if diff > 0.01:  # Small threshold to ignore noise
                                    movements_right += 1
                                elif diff < -0.01:
                                    movements_left += 1

                            for i in range(1, len(self.thumb_history)):
                                diff = self.thumb_history[i] - self.thumb_history[i-1]
                                if diff > 0.01:
                                    thumb_right += 1
                                elif diff < -0.01:
                                    thumb_left += 1

                            # Determine dominant direction
                            total_movements = movements_right + movements_left
                            total_thumb_movements = thumb_right + thumb_left

                            if total_movements > 0 and total_thumb_movements > 0:
                                right_ratio = movements_right / total_movements
                                left_ratio = movements_left / total_movements
                                thumb_right_ratio = thumb_right / total_thumb_movements
                                thumb_left_ratio = thumb_left / total_thumb_movements

                                # Check if thumb and fingers are moving in the same direction
                                # Be more lenient with left swipes for thumb agreement
                                thumb_finger_agreement_right = (right_ratio > 0.5 and thumb_right_ratio > 0.5)
                                thumb_finger_agreement_left = (left_ratio > 0.45 and thumb_left_ratio > 0.45)

                                # For fast/large swipes, be more lenient with consistency
                                # For slower swipes, require more consistency
                                is_fast_swipe = abs(total_movement) > self.fast_swipe_threshold

                                # Different consistency requirements for left vs right
                                # Left swipes get lower threshold (more sensitive)
                                required_consistency_left = 0.45 if is_fast_swipe else 0.55
                                required_consistency_right = 0.5 if is_fast_swipe else 0.6

                                # Detect swipe based on total movement AND consistency AND thumb agreement
                                if (total_movement > self.swipe_threshold_right and
                                    right_ratio > required_consistency_right and
                                    thumb_finger_agreement_right):
                                    gesture = 'right'
                                    self.last_gesture_time = current_time
                                    self.gesture_started = False
                                    self.finger_licked = False  # Reset after use
                                    self.position_history.clear()
                                    self.thumb_history.clear()
                                    swipe_type = "FAST" if is_fast_swipe else "normal"
                                    print(f"[GESTURE] RIGHT {swipe_type} swipe (movement: {total_movement:.3f}, thumb: {thumb_movement:.3f}, agreement: YES)")
                                    cv2.putText(frame, "PAGE TURN ->", (10, 30),
                                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                                elif (total_movement < -self.swipe_threshold_left and
                                      left_ratio > required_consistency_left and
                                      thumb_finger_agreement_left):
                                    gesture = 'left'
                                    self.last_gesture_time = current_time
                                    self.gesture_started = False
                                    self.finger_licked = False  # Reset after use
                                    self.position_history.clear()
                                    self.thumb_history.clear()
                                    swipe_type = "FAST" if is_fast_swipe else "normal"
                                    print(f"[GESTURE] LEFT {swipe_type} swipe (movement: {total_movement:.3f}, thumb: {thumb_movement:.3f}, agreement: YES)")
                                    cv2.putText(frame, "<- PAGE TURN", (10, 30),
                                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                        # Reset if movement is too slow (more than max_frames without trigger)
                        if self.movement_frames > self.max_frames:
                            self.gesture_started = False
                            self.position_history.clear()
                            self.thumb_history.clear()
                else:
                    # Reset gesture tracking if not ready
                    self.gesture_started = False
                    self.position_history.clear()
                    self.thumb_history.clear()

                # Show status indicator
                if not self.finger_licked:
                    cv2.putText(frame, "Lick finger to enable page turn", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                elif self.finger_licked and not currently_at_lips:
                    # Show remaining time when ready to swipe
                    time_remaining = self.lick_timeout - (current_time - self.lick_time)
                    if time_remaining > 0:
                        cv2.putText(frame, f"Ready for {time_remaining:.1f}s", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Show current position indicator
                cv2.circle(frame,
                          (int(current_x * frame.shape[1]), int(middle_finger_tip.y * frame.shape[0])),
                          10, (255, 0, 0), -1)
        else:
            # No hand detected, reset gesture tracking
            self.gesture_started = False
            self.start_x = None

        return gesture, frame
    
    def reset(self):
        """Reset gesture detection state"""
        self.start_x = None
        self.gesture_started = False
        self.movement_frames = 0
        self.last_gesture_time = 0
        self.position_history.clear()
        self.thumb_history.clear()
        self.finger_licked = False