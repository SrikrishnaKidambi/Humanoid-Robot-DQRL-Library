import numpy as np
import cv2
import mediapipe as mp
from .keyPointExtraction import PoseExtractor
from ultralytics import YOLO

def select_main_skeleton_multiple(extractor,image,save_path):
    """
    Try out multiple iterations of drawing skeletons for multiple people in the image and pick the one with more area
    """

    # use yolo 8 for detecting multiple people in the image

    yolo_model = YOLO("yolov8m.pt") # it is faster and small model

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    # if image.shape[-1] == 3:
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # detect all the people
    results = yolo_model.predict(source=image, verbose=False)
    boxes = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls = int(box.cls)
            if cls == 0:  # class 0 = 'person'
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))

    if not boxes:
        print("No person detected")
        return None
    
    main_box = max(boxes,key= lambda b: (b[2]-b[0])*(b[3]-b[1]))

    #extract only the max bounded box guy
    x1,y1,x2,y2 = main_box
    person_crop = image[y1:y2,x1:x2]

    skeleton = extractor.extract_keypoints(person_crop)
    if skeleton is None:
        print("Unable to find the pose")
        return None
    
    # adjusting the coordinates in reference to the original image
    skeleton[:,0] = (x1 + skeleton[:,0]*(x2-x1))/image.shape[1]
    skeleton[:,1] = (y1 + skeleton[:,1]*(y2-y1))/image.shape[0]

    extractor.draw_skeleton(image,skeleton,save_path)
    return skeleton


    # skeletons = []
    # for i in range(iterations):
    #     skel = extractor.extract_keypoints(image)
    #     if len(skel) > 0:
    #         skeletons.append(skel)

    # if not skeletons:
    #     return None
    
    # max_area, main_skel = 0 , None

    # for skel in skeletons:
    #     x,y = skel[:,0], skel[:,1]
    #     area = (x.max()-x.min()) * (y.max() - y.min())
    #     if area > max_area:
    #         max_area=area
    #         main_skel = skel
    # extractor.draw_skeleton(image,main_skel,save_path)
    # return main_skel



def compute_joint_angles(skeleton):
    """
    Computes the 9 relative joint angles (in radians) needed by the simulation.
    """
    
    # Helper function to get a vector (x, y) from two keypoint indices
    def get_vec(p1_idx, p2_idx):
        p1 = skeleton[p1_idx][:2]
        p2 = skeleton[p2_idx][:2]
        return p2 - p1

    # Helper function to get the signed angle between two vectors
    def angle_between(v1, v2):
        # Calculates the angle from v1 to v2
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        return angle2 - angle1

    # Limb vectors
    v_spine_up = get_vec(8, 1)     # MidHip -> Neck (our "up" reference)
    v_spine_down = -v_spine_up     # A vector pointing "down" the spine
    
    v_r_thigh = get_vec(9, 10)     # R Hip -> R Knee
    v_r_calf = get_vec(10, 11)     # R Knee -> R Ankle
    
    v_l_thigh = get_vec(12, 13)    # L Hip -> L Knee
    v_l_calf = get_vec(13, 14)     # L Knee -> L Ankle
    
    v_r_upper_arm = get_vec(2, 3)  # R Shoulder -> R Elbow
    v_r_forearm = get_vec(3, 4)    # R Elbow -> R Wrist
    
    v_l_upper_arm = get_vec(5, 6)  # L Shoulder -> L Elbow
    v_l_forearm = get_vec(6, 7)    # L Elbow -> L Wrist
    
    v_vertical_up = np.array([0, -1]) # Y=0 is top, so [0, -1] is UP

    # --- Calculate Relative Joint Angles ---
    
    # 1. R Hip
    r_hip_angle = angle_between(v_spine_down, v_r_thigh)
    # 2. R Knee
    r_knee_angle = angle_between(v_r_thigh, v_r_calf)
    # 3. L Hip
    l_hip_angle = angle_between(v_spine_down, v_l_thigh)
    # 4. L Knee
    l_knee_angle = angle_between(v_l_thigh, v_l_calf)
    
    # 5. R Shoulder
    r_shoulder_angle = angle_between(v_spine_down, v_r_upper_arm)
    # 6. R Elbow (No negative sign)
    r_elbow_angle = angle_between(v_r_upper_arm, v_r_forearm)
    
    # 7. L Shoulder (Mirrored axis)
    l_shoulder_angle = -angle_between(v_spine_down, v_l_upper_arm)
    # 8. L Elbow (Mirrored axis)
    l_elbow_angle = -angle_between(v_l_upper_arm, v_l_forearm)
    
    # 9. Spine
    spine_angle = angle_between(v_vertical_up, v_spine_up)

    angles = np.array([
        r_hip_angle, r_knee_angle,
        l_hip_angle, l_knee_angle,
        r_shoulder_angle, r_elbow_angle,
        l_shoulder_angle, l_elbow_angle,
        spine_angle
    ], dtype=np.float32)

    return angles

# def compute_joint_angles(skeleton):
#     """
#     Compute simple 2D joint angles (in radians) for main joints using atan2
#     """
#     # gives the limbs of the body as joints
#     joint_pairs = [
#         (9,10), # R Hip -> R Knee
#         (10,11), # R Knee -> R Ankle
#         (12,13), # L Hip -> L Knee
#         (13,14), # L Knee -> L Ankle
#         (2,3), # R shoulder -> R Elbow
#         (3,4), # R Elbow -> R Wrist
#         (5,6), # L shoulder -> L Elbow
#         (6,7), # L Elbow -> L wrist
#         (8,1) # MidHip -> Neck (spine)
#     ]
#     angles=[] #computed angles are stored here

#     for node1, node2 in joint_pairs:
#         node1, node2 = skeleton[node1][:2], skeleton[node2][:2] # this gives the x and y coordinates for two nodes of the limb
#         vec = node2 - node1 #(x2-x1,y2-y1)
#         angle = np.arctan2(vec[1],vec[0]) #angle wrt to x axis
#         # angle =0 vector straight right
#         # angle = pi/2 vector straight up
#         # angle = pi vector straight left
#         # angle = -pi/2 vector straing down

#         angles.append(angle)

#     return np.array(angles)

