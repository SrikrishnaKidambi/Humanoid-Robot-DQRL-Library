# import os

# current_dir = os.path.dirname(__file__)
# image_path = os.path.join(current_dir, "Person.jpeg")

from humanoid_library import load_image,preprocess_image
from humanoid_library import PoseExtractor, select_main_skeleton_multiple,compute_joint_angles

image1_path = "People.jpg"
image2_path = "Person.jpeg"
save1_path = "Output1.jpg"
save2_path = "Output2.jpg"

# load and process the image
image1 = load_image(image1_path)
image2 = load_image(image2_path)
image1r = preprocess_image(image1,(640,480))
image2r = preprocess_image(image2,(640,480))

# instantiate the pose extractor class
extractor = PoseExtractor()

main_skel1 = select_main_skeleton_multiple(extractor,image1r,save1_path)
main_skel2 = select_main_skeleton_multiple(extractor,image2r,save2_path)

if main_skel1 is None:
    print("No person in image1")
else:
    angles1 = compute_joint_angles(main_skel1)
    print("The joint angles are: ",angles1)

if main_skel2 is None:
    print("No person in image2")
else:
    angles2 = compute_joint_angles(main_skel2)
    print("The joint angles are: ",angles2)


# img = load_image("People.jpg")
# img_ready = preprocess_image(img,(368,368))
# print(img_ready.shape)
# extractor = PoseExtractor()
# keypoints = extractor.extract_keypoints("People.jpg",True,"output_pose1.jpg")
# print(keypoints.shape)
# print(keypoints)