import cv2
import numpy as np
import math

full_path  = "/home/anyone/Repositories/test_bed_RL_robot/robot_rotation_v2_full_aruco/camera_calibration"
matrix     = np.loadtxt((full_path + "/matrix.txt"))
distortion = np.loadtxt((full_path + "/distortion.txt"))


class Vision:
    def __init__(self):
        # Aruco Dictionary
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.arucoParams = cv2.aruco.DetectorParameters_create()

        self.markerSize = 18  # size of the aruco marker millimeters

        self.camera = cv2.VideoCapture(2)  # open the camera
        #self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        #self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

        self.robot_marks_id = [0, 1, 2, 3, 4, 5, 6]  # the id for each market in the robot
        self.vision_flag_status = False

    def get_camera_image(self):
        ret, frame = self.camera.read()
        if ret:
            return frame

    def get_index_id(self, aruco_id, id_detected):
        for id_index, id in enumerate(id_detected):
            if id == aruco_id:
                return id_index
        return -1

    def isclose(self, x, y, rtol=1.e-5, atol=1.e-8):
        return abs(x - y) <= atol + rtol * abs(y)

    def calculate_euler_angles(self, R):
        """
        From a paper by Gregory G. Slabaugh (undated),
        "Computing Euler angles from a rotation matrix
        """
        phi = 0.0
        if self.isclose(R[2, 0], -1.0):
            theta = math.pi / 2.0
            psi = math.atan2(R[0, 1], R[0, 2])
        elif self.isclose(R[2, 0], 1.0):
            theta = -math.pi / 2.0
            psi = math.atan2(-R[0, 1], -R[0, 2])
        else:
            theta = -math.asin(R[2, 0])
            cos_theta = math.cos(theta)
            psi = math.atan2(R[2, 1] / cos_theta, R[2, 2] / cos_theta)
            phi = math.atan2(R[1, 0] / cos_theta, R[0, 0] / cos_theta)
        return psi, theta, phi

    def get_angle(self, rot):
        rotation_matrix, _ = cv2.Rodrigues(rot)
        psi, theta, phi = self.calculate_euler_angles(rotation_matrix)
        phi = math.degrees(phi)
        return phi

    def calculate_marker_pose(self, goal_angle):
        # get image from camera
        image = self.get_camera_image()

        # Detect Aruco markers, corners and IDs
        (corners, IDs, rejected) = cv2.aruco.detectMarkers(image, self.arucoDict, parameters=self.arucoParams)
        cv2.aruco.drawDetectedMarkers(image, corners, borderColor=(0, 0, 255))
        # rotation and translation vector w.r.t camera
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.markerSize, matrix, distortion)

        try:
            if len(IDs) == 7:
                # Get the right index for each id of the detected markers
                for id_marker in self.robot_marks_id:
                    index = self.get_index_id(id_marker, IDs)
                    if id_marker == 0:
                        joint_0_arm_1_marker_index = index

                    if id_marker == 1:
                        joint_0_arm_2_marker_index = index

                    if id_marker == 2:
                        joint_1_arm_1_marker_index = index

                    if id_marker == 3:
                        joint_1_arm_2_marker_index = index

                    if id_marker == 4:
                        end_arm_1_marker_index = index

                    if id_marker == 5:
                        end_arm_2_marker_index = index

                    if id_marker == 6:
                        cylinder_marker_index = index

                joint_0_arm_1_location = tvec[joint_0_arm_1_marker_index][0][:-1]  # I am not considering z axis here
                joint_0_arm_2_location = tvec[joint_0_arm_2_marker_index][0][:-1]

                joint_1_arm_1_location = tvec[joint_1_arm_1_marker_index][0][:-1]
                joint_1_arm_2_location = tvec[joint_1_arm_2_marker_index][0][:-1]

                end_arm_1_location = tvec[end_arm_1_marker_index][0][:-1]
                end_arm_2_location = tvec[end_arm_2_marker_index][0][:-1]

                cylinder_location = tvec[cylinder_marker_index][0][:-1]
                cylinder_angle = np.array([self.get_angle(rvec[cylinder_marker_index][0])])

                goal_angle = np.array([goal_angle])

                state_space = (joint_0_arm_1_location, joint_0_arm_2_location,
                               joint_1_arm_1_location, joint_1_arm_2_location,
                               end_arm_1_location, end_arm_2_location,
                               cylinder_location, cylinder_angle, goal_angle)

                self.vision_flag_status = True
                return state_space, image, self.vision_flag_status

            else:
                print("not all aruco markers no detected")
                self.vision_flag_status = False
                return 0, image, self.vision_flag_status

        except:
            print("camara obstruction")
            self.vision_flag_status = False
            return 0, image, self.vision_flag_status
