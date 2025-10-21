import numpy as np
import cv2
import mediapipe as mp
from .keyPointExtraction import PoseExtractor

def select_main_skeleton_multiple(extractor,image,save_path, iterations=5):
    """
    Try out multiple iterations of drawing skeletons for multiple people in the image and pick the one with more area
    """
    skeletons = []
    for i in range(iterations):
        skel = extractor.extract_keypoints(image)
        if len(skel) > 0:
            skeletons.append(skel)

    if not skeletons:
        return None
    
    max_area, main_skel = 0 , None

    for skel in skeletons:
        x,y = skel[:,0], skel[:,1]
        area = (x.max()-x.min()) * (y.max() - y.min())
        if area > max_area:
            max_area=area
            main_skel = skel
    extractor.draw_skeleton(image,main_skel,save_path)
    return main_skel
    
def compute_joint_angles(skeleton):
    """
    Compute simple 2D joint angles (in radians) for main joints using atan2
    """
    # gives the limbs of the body as joints
    joint_pairs = [
        (9,10), # R Hip -> R Knee
        (10,11), # R Knee -> R Ankle
        (12,13), # L Hip -> L Knee
        (13,14), # L Knee -> L Ankle
        (2,3), # R shoulder -> R Elbow
        (3,4), # R Elbow -> R Wrist
        (5,6), # L shoulder -> L Elbow
        (6,7), # L Elbow -> L wrist
        (8,1) # MidHip -> Neck (spine)
    ]
    angles=[] #computed angles are stored here

    for node1, node2 in joint_pairs:
        node1, node2 = skeleton[node1][:2], skeleton[node2][:2] # this gives the x and y coordinates for two nodes of the limb
        vec = node2 - node1 #(x2-x1,y2-y1)
        angle = np.arctan2(vec[1],vec[0]) #angle wrt to x axis
        # angle =0 vector straight right
        # angle = pi/2 vector straight up
        # angle = pi vector straight left
        # angle = -pi/2 vector straing down

        angles.append(angle)

    return np.array(angles)

