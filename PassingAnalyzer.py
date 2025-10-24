from collections import deque
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import math
import csv
import cv2

yolo_model = YOLO("yolov8s.pt")
mp_pose = mp.solutions.pose

WINDOW_SIZE = 10
FOOT_DISTANCE_THRESHOLD = 20
KNEE_ANGLE_THRESHOLD = 15
BALL_MOVEMENT_THRESHOLD = 20
STANDARD_BALL_DIAMETER_CM = 22
SCORE_THRESHOLD = 0.65

def get_angle(a, b, c):
    BA = np.array([a[0] - b[0], -(a[1] - b[1])])
    BC = np.array([c[0] - b[0], -(c[1] - b[1])])
    cross = BA[0] * BC[1] - BA[1] * BC[0]
    dot = BA[0] * BC[0] + BA[1] * BC[1]
    angle = math.degrees(math.atan2(cross, dot))
    angle = (angle + 360) % 360
    return angle

def get_arc_angles(b, c1, c2):
    a_x_axis = np.array([b[0] + 100, b[1]])
    return (get_angle(a_x_axis, b, c1), get_angle(a_x_axis, b, c2))

def get_min_distance_frame(history):
    valid_entries = [x for x in history if x['shooting_foot_distance'] is not None]
    if not valid_entries:
        return None
    return min(valid_entries, key=lambda x: x['shooting_foot_distance'])

def get_nearest_surface(landmarks, stationary_foot, cords):
    (x1, y1, x2, y2) = cords
    mids = [(x1, (y1 + y2) / 2), (x2, (y1 + y2) / 2)]
    candidates_back = mids    # Potentially increased
    best_point = min(candidates_back, key=lambda p: math.dist(landmarks[stationary_foot], p))
    ball_surface = tuple(map(int, best_point))
    return ball_surface

def put_text(frame, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, text_color=(0,0,0), bg_color=(255,255,255), thickness=2):
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x, y - th - baseline), (x + tw, y + baseline), bg_color, -1)
    cv2.putText(frame, text, org, font, font_scale, text_color, thickness, cv2.LINE_AA)

def draw_points_and_lines(frame, pts, line_color=(0, 255, 0), point_color=(0, 128, 255)):
    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i + 1], line_color, 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 3, point_color, 3, cv2.LINE_AA)
        cv2.circle(frame, pt, 2, (0, 0, 0), -1, cv2.LINE_AA)

def draw_vector(frame, pts):
    if len(pts) == 2:
        cv2.arrowedLine(frame, pts[0], pts[1], (255, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(frame, pts[0], 3, (255, 255, 255), 2, cv2.LINE_AA)

def draw_rainbow(frame, trail):
    if len(trail) < 2:
        return frame
    overlay = frame.copy()
    for i in range(1, len(trail)):
        pt1 = trail[i-1]
        pt2 = trail[i]
        hue = int(180 * i / len(trail))
        color = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0][0]
        color = tuple(int(c) for c in bgr)
        cv2.line(overlay, pt1, pt2, color, 6, cv2.LINE_AA)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return frame

def visualize_side_frame(frame, landmarks, stationary_foot, ball_surface, l_knee_angle, r_knee_angle, trunk_angle):

    # draw_points_and_lines(frame, [landmarks[stationary_foot], ball_surface], line_color=(0, 128, 255))
    draw_points_and_lines(frame, [landmarks["l_hip"], landmarks["l_knee"], landmarks["l_ankle"]])
    draw_points_and_lines(frame, [landmarks["r_hip"], landmarks["r_knee"], landmarks["r_ankle"]])

    draw_vector(frame, (landmarks["l_heel"], landmarks["l_foot"]))
    draw_vector(frame, (landmarks["r_heel"], landmarks["r_foot"]))

    put_text(frame, f"{round(l_knee_angle)}", (landmarks["l_knee"][0], landmarks["l_knee"][1] - 20))
    put_text(frame, f"{round(r_knee_angle)}", (landmarks["r_knee"][0] - 80, landmarks["r_knee"][1] + 40))

    l_knee_arc = get_arc_angles(landmarks["l_knee"], landmarks["l_hip"], landmarks["l_ankle"])
    r_knee_arc = get_arc_angles(landmarks["r_knee"], landmarks["r_hip"], landmarks["r_ankle"])

    if "r_foot" == stationary_foot:
        draw_points_and_lines(frame, [landmarks["l_shoulder"], landmarks["l_hip"], landmarks["l_knee"]])
        put_text(frame, f"{round(trunk_angle)}", (landmarks["l_hip"][0] - 50, landmarks["l_hip"][1]))
        trunk_arc = get_arc_angles(landmarks["l_hip"], landmarks["l_shoulder"], landmarks["l_knee"])
    else:
        draw_points_and_lines(frame, [landmarks["r_shoulder"], landmarks["r_hip"], landmarks["r_knee"]])
        put_text(frame, f"{round(trunk_angle)}", (landmarks["r_hip"][0] + 50, landmarks["r_hip"][1]))
        trunk_arc = get_arc_angles(landmarks["r_hip"], landmarks["r_shoulder"], landmarks["r_knee"])

    overlay = frame.copy()
    cv2.ellipse(overlay, landmarks["l_knee"], (40, 40), 0, int(-l_knee_arc[0]), int(-l_knee_arc[1]), (0, 128, 255), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    cv2.ellipse(frame, landmarks["l_knee"], (40, 40), 0, int(-l_knee_arc[0]), int(-l_knee_arc[1]), (0, 128, 255), 2, cv2.LINE_AA)

    overlay = frame.copy()
    cv2.ellipse(overlay, landmarks["r_knee"], (40, 40), 0, int(-r_knee_arc[0]), int(-r_knee_arc[1]), (0, 128, 255), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    cv2.ellipse(frame, landmarks["r_knee"], (40, 40), 0, int(-r_knee_arc[0]), int(-r_knee_arc[1]), (0, 128, 255), 2, cv2.LINE_AA)

    overlay = frame.copy()
    if "r_foot" == stationary_foot:
        cv2.ellipse(overlay, landmarks["l_hip"], (40, 40), 0, int(-trunk_arc[0]), int(-trunk_arc[1] + 360), (0, 128, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.ellipse(frame, landmarks["l_hip"], (40, 40), 0, int(-trunk_arc[0]), int(-trunk_arc[1] + 360), (0, 128, 255), 2, cv2.LINE_AA)
    else:
        cv2.ellipse(overlay, landmarks["r_hip"], (40, 40), 0, int(-trunk_arc[0]), int(-trunk_arc[1] + 360), (0, 128, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.ellipse(frame, landmarks["r_hip"], (40, 40), 0, int(-trunk_arc[0]), int(-trunk_arc[1] + 360), (0, 128, 255), 2, cv2.LINE_AA)

def visualize_back_frame(frame, landmarks, ball_surface, stationary_foot, st_foot_ball_dis_back):
    draw_points_and_lines(frame, [ball_surface, landmarks[stationary_foot]])
    put_text(frame, f"{st_foot_ball_dis_back} CM", (landmarks[stationary_foot][0]+20, landmarks[stationary_foot][1]+50))

def detect_ball(frame):
    persons, balls = [], []
    pair, min_dist = None, float('inf')
    
    result = yolo_model.predict(frame, verbose=False)[0]
    
    for box, cls_id, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
        if conf < SCORE_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(cls_id)

        if cls_id == 0:
            persons.append((x1, y1, x2, y2))
        elif cls_id == 32:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            w, h = x2 - x1, y2 - y1
            d = max(w, h)
            cpp = STANDARD_BALL_DIAMETER_CM / d
            balls.append(((cx, cy), (x1, y1, x2, y2), cpp))

    for (x1, y1, x2, y2) in persons:
        px, py = (x1 + x2) // 2, (y1 + y2) // 2
        for (bc, bbox, cpp) in balls:
            bx, by = bc
            dist = (px - bx) ** 2 + (py - by) ** 2
            if dist < min_dist:
                min_dist = dist
                pair = (bc, cpp, bbox, (x1, y1, x2, y2))
    if pair:
        bbox = pair[2]
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    return pair if pair else (None, None, None, None)

def detect_landmarks(rgb_frame, pose, w, h):
    pose_result = pose.process(rgb_frame)
    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks.landmark
        l_shoulder = (int(landmarks[11].x * w), int(landmarks[11].y * h))
        r_shoulder = (int(landmarks[12].x * w), int(landmarks[12].y * h))
        l_hip = (int(landmarks[23].x * w), int(landmarks[23].y * h))
        r_hip = (int(landmarks[24].x * w), int(landmarks[24].y * h))
        l_knee = (int(landmarks[25].x * w), int(landmarks[25].y * h))
        r_knee = (int(landmarks[26].x * w), int(landmarks[26].y * h))
        l_ankle = (int(landmarks[27].x * w), int(landmarks[27].y * h))
        r_ankle = (int(landmarks[28].x * w), int(landmarks[28].y * h))
        l_heel = (int(landmarks[29].x * w), int(landmarks[29].y * h))
        r_heel = (int(landmarks[30].x * w), int(landmarks[30].y * h))
        l_foot = (int(landmarks[31].x * w), int(landmarks[31].y * h))
        r_foot = (int(landmarks[32].x * w), int(landmarks[32].y * h))
        return {"l_shoulder": l_shoulder, "r_shoulder": r_shoulder, "l_hip": l_hip, "r_hip": r_hip, "l_knee": l_knee, "r_knee": r_knee,
                "l_ankle": l_ankle, "r_ankle": r_ankle, "l_heel": l_heel, "r_heel": r_heel, "l_foot": l_foot, "r_foot": r_foot}
    return None

def detect_shooting_leg(frame, ball_center, person_box, landmarks):
    if not (ball_center and landmarks and person_box):
        return "r_foot", None
    
    x1, y1, x2, y2 = person_box
    person_center_x = (x1 + x2) // 2
    bx = ball_center[0]

    if bx < person_center_x:
        side = "l"
    elif bx > person_center_x:
        side = "r"
    else:
        return None, None

    foot, hip, knee, ankle = f"{side}_foot", f"{side}_hip", f"{side}_knee", f"{side}_ankle"
    if landmarks:
        pts = landmarks[hip], landmarks[knee], landmarks[ankle]
        angle = get_angle(*pts)
        draw_points_and_lines(frame, pts)
        return foot, angle

    return foot, None

def process_side(dis_frame_side, pose_side, trail_side, w_side, h_side):
    shooting_foot_distance = moving_leg_angle = ball_center_side = moving_foot = None
    rgb_frame_side = cv2.cvtColor(dis_frame_side, cv2.COLOR_BGR2RGB)
    dis_frame_side = draw_rainbow(dis_frame_side, trail_side)
    (ball_center_side, CpP_side, cords_side, person_box) = detect_ball(dis_frame_side)
    
    if ball_center_side:
        trail_side.append(ball_center_side)
    
    landmarks_side = detect_landmarks(rgb_frame_side, pose_side, w_side, h_side)

    if landmarks_side:
        draw_vector(dis_frame_side, (landmarks_side["l_heel"], landmarks_side["l_foot"]))
        draw_vector(dis_frame_side, (landmarks_side["r_heel"], landmarks_side["r_foot"]))
        moving_foot, moving_leg_angle = detect_shooting_leg(dis_frame_side, ball_center_side, person_box, landmarks_side)
        
        if moving_foot and ball_center_side and CpP_side and cords_side:
            if moving_foot == "l_foot": moving_idx = landmarks_side["l_foot"]  
            else: moving_idx = landmarks_side["r_foot"]
            
            shooting_foot_distance = math.dist(moving_idx, ball_center_side) * CpP_side
            draw_points_and_lines(dis_frame_side, [ball_center_side, moving_idx])

    stationary_foot = "r_foot" if moving_foot == "l_foot" else "l_foot"
    return dis_frame_side, shooting_foot_distance, moving_leg_angle, ball_center_side, stationary_foot


def detect_ball_person_for_back_frame(frame):
    global last_valid_box_back, last_valid_center_back, last_valid_cpp_back
    global last_right_edge, last_left_edge, frames_since_detection_back
    
    yolo_result = yolo_model.predict(frame, verbose=False)[0]
    found_persons = []
    found_ball = None
    found_ball_box = None
    found_cpp = None
    valid_ball_detected = False
    
    for box, cls_id, score in zip(yolo_result.boxes.xyxy, yolo_result.boxes.cls, yolo_result.boxes.conf):
        cls_id = int(cls_id)
        x1, y1, x2, y2 = map(int, box)
        if score < SCORE_THRESHOLD:
            continue
            
        if cls_id == 0:
            found_persons.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Person {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        elif cls_id == 32:
            width = x2 - x1
            height = y2 - y1
            ball_diameter = (width + height) / 2
            if 10 < ball_diameter < 100 and not math.isnan(ball_diameter):
                frames_since_detection_back = 0
            
                if ball_diameter > 0:
                    found_cpp = STANDARD_BALL_DIAMETER_CM / ball_diameter
                    last_valid_cpp_back = found_cpp
                
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                found_ball = center
                found_ball_box = (x1, y1, x2, y2)
                
                last_valid_center_back = found_ball
                last_valid_box_back = found_ball_box
                last_right_edge = (x2, (y1 + y2) // 2)
                last_left_edge = (x1, (y1 + y2) // 2)
                
                valid_ball_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Ball {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
 
        
        found_ball = last_valid_center_back
        found_ball_box = last_valid_box_back
        found_cpp = last_valid_cpp_back
        
        if last_valid_box_back:
            x1, y1, x2, y2 = last_valid_box_back
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"Ball (Est. {frames_since_detection_back}f)", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    return found_ball, found_cpp, found_ball_box, found_persons

def process_shoot_back(dis_frame_back, pose_back, trail_back, stationary_foot, w_back, h_back):
    global last_right_edge, last_left_edge, frames_since_detection_back, last_valid_cpp_back
    
    rgb_frame_back = cv2.cvtColor(dis_frame_back, cv2.COLOR_BGR2RGB)
    ball_center_back, CpP_back, cords_back, _ = detect_ball_person_for_back_frame(dis_frame_back)
    
    if ball_center_back is not None:
        trail_back.append(ball_center_back)
    
    landmarks_back = detect_landmarks(rgb_frame_back, pose_back, w_back, h_back)
    
    if landmarks_back and last_valid_cpp_back is not None:
        if stationary_foot == "l_foot" and "l_ankle" in landmarks_back and last_right_edge:
            draw_points_and_lines(dis_frame_back, [landmarks_back["l_ankle"], last_right_edge])
            l_ankle_ball_right_dis = math.dist(landmarks_back["l_ankle"], last_right_edge) * last_valid_cpp_back
            corrected_distance = max(0, l_ankle_ball_right_dis - (STANDARD_BALL_DIAMETER_CM / 2))
            
            distance_text = f"{corrected_distance:.1f} CM"
            if frames_since_detection_back > 0: 
                distance_text += " (Est.)"
            
            #put_text_with_bg(dis_frame_back, distance_text, landmarks_back["l_ankle"])
            
        elif stationary_foot == "r_foot" and "r_ankle" in landmarks_back and last_left_edge:
            draw_points_and_lines(dis_frame_back, [landmarks_back["r_ankle"], last_left_edge])
            r_ankle_ball_left_dis = math.dist(landmarks_back["r_ankle"], last_left_edge) * last_valid_cpp_back
            corrected_distance = max(0, r_ankle_ball_left_dis - (STANDARD_BALL_DIAMETER_CM / 2))
            
            distance_text = f"{corrected_distance:.1f} CM"
            if frames_since_detection_back > 0:  
                distance_text += " (Est.)"
            
            #put_text_with_bg(dis_frame_back, distance_text, landmarks_back["r_ankle"])
            
        #cv2.putText(dis_frame_back,stationary_foot, (10, h_back - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
         #       (0, 255, 0) , 2)
    
    return dis_frame_back

def detect_frame(history):
    if len(history) < WINDOW_SIZE:
        return None, None, None
    
    first, last = history[0], history[-1]
    
    distance_change = last['shooting_foot_distance'] - first['shooting_foot_distance'] if last['shooting_foot_distance'] and first['shooting_foot_distance'] else 0
    angle_change = last['moving_leg_angle'] - first['moving_leg_angle'] if last['moving_leg_angle'] and first['moving_leg_angle'] else 0
    ball_movement = math.dist(last['ball_center'], first['ball_center']) if last['ball_center'] and first['ball_center'] else 0
    
    if (-distance_change >= FOOT_DISTANCE_THRESHOLD and angle_change >= KNEE_ANGLE_THRESHOLD and ball_movement >= BALL_MOVEMENT_THRESHOLD):
        contact_frame_data = get_min_distance_frame(history)
        if contact_frame_data:
            return contact_frame_data['frame_side'], contact_frame_data['frame_back'], True
    return None, None, False

def display_frame(frame_side, frame_back, side_dis_width, side_dis_height, back_dis_width, back_dis_height, history, stationary_foot):
    detection = {}
    l_knee_angle = r_knee_angle = trunk_angle = st_foot_ball_dis_back = None

    pose_side_static = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    pose_back_static = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    try:
        if frame_side is not None:
            (ball_center_side, CpP_side, cords_side, _) = detect_ball(frame_side)
            frame_side_rgb = cv2.cvtColor(frame_side, cv2.COLOR_BGR2RGB)
            h_side, w_side, _ = frame_side.shape
            landmarks_side = detect_landmarks(frame_side_rgb, pose_side_static, w_side, h_side)
            if landmarks_side and cords_side:
                l_knee_angle = get_angle(landmarks_side["l_hip"], landmarks_side["l_knee"], landmarks_side["l_ankle"])
                r_knee_angle = get_angle(landmarks_side["r_hip"], landmarks_side["r_knee"], landmarks_side["r_ankle"])
                if "r_foot" == stationary_foot:
                    trunk_angle = get_angle(landmarks_side["l_knee"], landmarks_side["l_hip"], landmarks_side["l_shoulder"])
                else:
                    trunk_angle = get_angle(landmarks_side["r_knee"], landmarks_side["r_hip"], landmarks_side["r_shoulder"])
                ball_surface_side = get_nearest_surface(landmarks_side, stationary_foot, cords_side)
                st_foot_ball_dis_side = math.dist(landmarks_side[stationary_foot], ball_surface_side) * CpP_side
                visualize_side_frame(frame_side, landmarks_side, stationary_foot, ball_surface_side, l_knee_angle, r_knee_angle, trunk_angle)

        if frame_back is not None:
            (ball_center_back, CpP_back, cords_back, _) = detect_ball(frame_back)
            frame_back_rgb = cv2.cvtColor(frame_back, cv2.COLOR_BGR2RGB)
            h_back, w_back, _ = frame_back.shape
            landmarks_back = detect_landmarks(frame_back_rgb, pose_back_static, w_back, h_back)
            if landmarks_back and cords_back:
                ball_surface_back = get_nearest_surface(landmarks_back, stationary_foot, cords_back)
                st_foot_ball_dis_back = round(math.dist(landmarks_back[stationary_foot], ball_surface_back) * CpP_back)
                visualize_back_frame(frame_back, landmarks_back, ball_surface_back, stationary_foot, st_foot_ball_dis_back)

        moving_foot = "Left" if stationary_foot == 'r_foot' else "Right"

        detection["Side Frame"] = frame_side
        detection["Back Frame"] = frame_back
        detection["Moving Foot"] = moving_foot
        detection["Left knee Angle"] = f"{round(l_knee_angle)} °" if l_knee_angle else F"Not detected"
        detection["Right Knee Angle"] = f"{round(r_knee_angle)} °" if r_knee_angle else F"Not detected"
        detection["Trunk Angle"] = f"{round(trunk_angle)} °" if trunk_angle else F"Not detected"
        detection["Ball–Stationary Foot Distance"] = f"{st_foot_ball_dis_back} CM" if st_foot_ball_dis_back else F"Not detected"

        history.clear()

    finally:
        pose_side_static.close()
        pose_back_static.close()
        return detection

def main(side_path, back_path):
    detections = []

    pose_side = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    pose_back = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    history = deque(maxlen=WINDOW_SIZE)
    trail_side = deque(maxlen=90)
    trail_back = deque(maxlen=90)

    cap_side = cv2.VideoCapture(side_path)
    cap_back = cv2.VideoCapture(back_path)

    if not cap_side.isOpened():
        raise FileNotFoundError(f"Could not open side video")
    if not cap_back.isOpened():
        raise FileNotFoundError(f"Could not open back video")

    side_fps = cap_side.get(cv2.CAP_PROP_FPS)
    side_orig_size = int(cap_side.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_side.get(cv2.CAP_PROP_FRAME_HEIGHT))
    side_dis_width = 700
    side_dis_height = int((side_dis_width / side_orig_size[0]) * side_orig_size[1])
    side_dis_size = side_dis_width, side_dis_height

    back_fps = cap_back.get(cv2.CAP_PROP_FPS)
    back_orig_size = int(cap_back.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_back.get(cv2.CAP_PROP_FRAME_HEIGHT))
    back_dis_width = 700
    back_dis_height = int((back_dis_width / back_orig_size[0]) * back_orig_size[1])
    back_dis_size = back_dis_width, back_dis_height

    try:
        while True:
            ret_side, frame_side = cap_side.read()
            ret_back, frame_back = cap_back.read()
            
            if not (ret_side and ret_back):
                break

            ball_center_side = shooting_foot_distance = moving_leg_angle = stationary_foot = None

            orig_frame_side = frame_side.copy()
            orig_frame_back = frame_back.copy()
            dis_frame_side = orig_frame_side.copy()
            dis_frame_back = orig_frame_back.copy()
            h_side, w_side, _ = dis_frame_side.shape
            h_back, w_back, _ = dis_frame_back.shape

            dis_frame_side, shooting_foot_distance, moving_leg_angle, ball_center_side, stationary_foot = process_side(dis_frame_side, pose_side, trail_side, w_side, h_side)
            dis_frame_back = process_back(dis_frame_back, pose_back, trail_back, stationary_foot, w_back, h_back)
            
            history.append({
                'frame_side': orig_frame_side, 
                'frame_back': orig_frame_back, 
                'moving_leg_angle': moving_leg_angle, 
                'shooting_foot_distance': shooting_foot_distance, 
                'ball_center': ball_center_side
            })

            contact_frame_side, contact_frame_back, flag = detect_frame(history)
            if flag:
                detection = display_frame(contact_frame_side, contact_frame_back, side_dis_width, side_dis_height, back_dis_width, back_dis_height, history, stationary_foot)
                detections.append(detection)
    
    finally:
        trail_side.clear()
        trail_back.clear()
        cap_side.release()
        cap_back.release()
        pose_side.close()
        pose_back.close()

    return detections
