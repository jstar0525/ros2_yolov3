from __future__ import division

# Python imports
import numpy as np
from pkg_resources import declare_namespace
import scipy.io as sio
import os, sys, cv2, time
from skimage.transform import resize
from torch.cuda import is_available

# ROS imports
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup

import std_msgs.msg
from std_msgs.msg import UInt8
from sensor_msgs.msg import Image
from geometry_msgs.msg import Polygon, Point32
from cv_bridge import CvBridge, CvBridgeError
from yolov3_ros_interfaces.msg import BoundingBox, BoundingBoxes

# Deep learning imports
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

<<<<<<< HEAD
from yolov3_ros.models.models import Darknet
from yolov3_ros.utils.utils import *
=======
from models import Darknet
from utils.utils import *
>>>>>>> afdd24e6001fb2a35ad60fe9353fe7af0b6f9fdf

# Detector manager class for YOLO
class DetectorManager(Node):

    def __init__(self):
        super().__init__('Detector')
        # Load weights parameter
        self.declare_parameter('weights_name', 'yolov3.weights')
        weights_name = self.get_parameter('weights_name').value
<<<<<<< HEAD
        self.weights_path = os.getenv('HOME') + '/model_cfg/' + weights_name
        self.get_logger().info(f'Found weights, loading {self.weights_path}')
=======
        self.weights_path = os.getenv('home') + '/model_cfg/' + weights_name
        self.get_logger().info("Found weights, loading %s", self.weights_path)
>>>>>>> afdd24e6001fb2a35ad60fe9353fe7af0b6f9fdf

        # Raise error if it cannot find the model
        if not os.path.isfile(self.weights_path):
            raise IOError(('{:s} not found.').format(self.weights_path))

        # Load parameters
        self.declare_parameter('subscribe_topic', '/image')
        self.subscribe_topic = self.get_parameter('subscribe_topic').value

        self.declare_parameter('confidence', 0.7)
        self.confidence = self.get_parameter('confidence').value

<<<<<<< HEAD
        self.declare_parameter('NMS', 0.3)
=======
        self.decaler_parameter('NMS', 0.3)
>>>>>>> afdd24e6001fb2a35ad60fe9353fe7af0b6f9fdf
        self.NMS = self.get_parameter('NMS').value

        self.declare_parameter('config_name', 'yolov3.cfg')
        config_name = self.get_parameter('config_name').value
<<<<<<< HEAD
        self.config_path = os.getenv('HOME') + '/model_cfg/' + config_name

        self.declare_parameter('classes_name', 'coco.names')
        classes_name = self.get_parameter('classes_name').value
        self.classes_path = os.getenv('HOME') + '/model_cfg/' + classes_name
=======
        self.config_path = os.getenv('home') + '/model_cfg/' + config_name

        self.declare_paremeter('classes_name', 'coco.names')
        classes_name = self.get_parameter('classes_name').value
        self.classes_path = os.getenv('home') + '/model_cfg/' + classes_name
>>>>>>> afdd24e6001fb2a35ad60fe9353fe7af0b6f9fdf

        self.declare_parameter('gpu_id', 0)
        self.gpu_id = self.get_parameter('gpu_id').value

        self.declare_parameter('image_size', 416)
        self.image_size = self.get_parameter('image_size').value

        self.declare_parameter('isVisualizing', True)
        self.isVisualizing = self.get_parameter('isVisualizing').value

        self.callback_group = ReentrantCallbackGroup()
<<<<<<< HEAD
        # self.add_on_set_parameters_callback(self.update_parameter) # after foxy
        # self.set_parameters_callback(self.update_parameter)
=======
        self.add_on_set_parameters_callback(self.update_parameter)
>>>>>>> afdd24e6001fb2a35ad60fe9353fe7af0b6f9fdf
        
        # Initialize width and height
        self.h = 0
        self.w = 0

        self.get_logger().info("config path: " + self.config_path)
<<<<<<< HEAD
        self.model = Darknet(self.config_path, self.image_size)
=======
        self.model = Darknet(self.config_path, image_size = self.image_size)
>>>>>>> afdd24e6001fb2a35ad60fe9353fe7af0b6f9fdf

        # Load net
        self.model.load_weights(self.weights_path)
        if torch.cuda.is_available():
            self.get_logger().info("CUDA available, use GPU")
            self.model.cuda()
        else:
            self.get_logger().info("CUDA not available, use CPU")
            # if CUDA not available, use CPU
            # self.checkpoint = torch.load(self.weights_path, map_location=torch.device('cpu'))
            # self.model.load_state_dict(self.checkpoint)
        self.model.eval() # Set in evaluation mode
        self.get_logger().info("Deep neural network loaded")

        # Load CvBridge
        self.bridge = CvBridge()

        # Load classes
        self.classes = load_classes(self.classes_path) # Extracts class labels from file
        self.classes_colors = {}
<<<<<<< HEAD
=======
        
        # Define subscribers
        self.image_sub = rclpy.Subscriber(self.image_topic, Image, self.imageCb, queue_size = 1, buff_size = 2**24)
>>>>>>> afdd24e6001fb2a35ad60fe9353fe7af0b6f9fdf

        # QoS
        self.declare_parameter('qos_depth', 10)
        qos_depth = self.get_parameter('qos_depth').value
        QOS_RKL10V = QoSProfile(
            reliability = QoSReliabilityPolicy.RELIABLE,
            history = QoSHistoryPolicy.KEEP_LAST,
            depth = qos_depth,
            durability = QoSDurabilityPolicy.VOLATILE)

        # Define subscriber
<<<<<<< HEAD
        self.image_sub = self.create_subscription(
=======
        self.image_sub = self.create_subcription(
>>>>>>> afdd24e6001fb2a35ad60fe9353fe7af0b6f9fdf
            Image,
            self.subscribe_topic,
            self.imageCb,
            QOS_RKL10V,
            callback_group = self.callback_group
        )

        # Define publishers
        self.detected_objects_topic = self.create_publisher(
            BoundingBoxes,
            'detected_objects_topic',
            QOS_RKL10V
        )
        self.published_image_topic = self.create_publisher(
            Image,
            'detected_objects_topic',
            QOS_RKL10V
        )
        self.get_logger().info("Launched node for object detection")

    def imageCb(self, data):
        # Convert the image to OpenCV
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)

        # Initialize detection results
        detection_results = BoundingBoxes()
        detection_results.header = data.header
        detection_results.image_header = data.header

        # Configure input
        input_img = self.imagePreProcessing(self.cv_image)

        # set image type
        if(torch.cuda.is_available()):
          input_img = Variable(input_img.type(torch.cuda.FloatTensor))
        else:
          input_img = Variable(input_img.type(torch.FloatTensor))

        # Get detections from network
        with torch.no_grad():
            detections = self.model(input_img)
            detections = non_max_suppression(detections, 80, self.confidence, self.NMS)
        
        # Parse detections
        if detections[0] is not None:
            for detection in detections[0]:
                # Get xmin, ymin, xmax, ymax, confidence and class
                xmin, ymin, xmax, ymax, _, conf, det_class = detection
                pad_x = max(self.h - self.w, 0) * (self.image_size/max(self.h, self.w))
                pad_y = max(self.w - self.h, 0) * (self.image_size/max(self.h, self.w))
                unpad_h = self.image_size-pad_y
                unpad_w = self.image_size-pad_x
                xmin_unpad = int(((xmin-pad_x//2)/unpad_w)*self.w)
                xmax_unpad = int(((xmax-xmin)/unpad_w)*self.w + xmin_unpad)
                ymin_unpad = int(((ymin-pad_y//2)/unpad_h)*self.h)
                ymax_unpad = int(((ymax-ymin)/unpad_h)*self.h + ymin_unpad)

                # Populate darknet message
                detection_msg = BoundingBox()
                detection_msg.xmin = xmin_unpad
                detection_msg.xmax = xmax_unpad
                detection_msg.ymin = ymin_unpad
                detection_msg.ymax = ymax_unpad
                detection_msg.probability = float(conf)
                detection_msg.class_id = self.classes[int(det_class)]

                # Append in overall detection message
                detection_results.bounding_boxes.append(detection_msg)

            # Publish detection results
            self.detected_objects_topic.publish(detection_results)

            # Visualize detection results
            if self.isVisualizing:
                self.VisualizeAndPublish(detection_results, self.cv_image)
        else:
            self.get_logger().info("No detections available, next image")
        return True
    

    def imagePreProcessing(self, img):
        # Extract image and shape
        img = np.ascontiguousarray(img)
        img = img.astype(float)
        height, width, channels = img.shape
        
        if (height != self.h) or (width != self.w):
            self.h = height
            self.w = width
            
            # Determine image to be used
            self.padded_image = np.zeros((max(self.h,self.w), max(self.h,self.w), channels)).astype(float)
            
        # Add padding
        if (self.w > self.h):
            self.padded_image[(self.w-self.h)//2 : self.h + (self.w-self.h)//2, :, :] = img
        else:
            self.padded_image[:, (self.h-self.w)//2 : self.w + (self.h-self.w)//2, :] = img
        
        # Resize and normalize
        input_img = resize(self.padded_image, (self.image_size, self.image_size, 3))/255.

        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))

        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()
        input_img = input_img[None]

        return input_img


    def VisualizeAndPublish(self, output, imgIn):
        # Copy image and visualize
        imgOut = np.ascontiguousarray(imgIn)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.6
        thickness = int(2)
        for index in range(len(output.bounding_boxes)):
            label = output.bounding_boxes[index].class_id
            x_p1 = output.bounding_boxes[index].xmin
            y_p1 = output.bounding_boxes[index].ymin
            x_p3 = output.bounding_boxes[index].xmax
            y_p3 = output.bounding_boxes[index].ymax
            confidence = output.bounding_boxes[index].probability

            # Find class color
            if label in self.classes_colors.keys():
                color = self.classes_colors[label]
            else:
                # Generate a new color if first time seen this label
                color = np.random.randint(0,255,3)
                self.classes_colors[label] = color
            
            # Create rectangle
            start_point = (int(x_p1), int(y_p1))
            end_point = (int(x_p3), int(y_p3))
            lineColor = (int(color[0]), int(color[1]), int(color[2]))

            cv2.rectangle(imgOut, start_point, end_point, lineColor, thickness)
            text = ('{:s}: {:.3f}').format(label,confidence)
            cv2.putText(imgOut, text, (int(x_p1), int(y_p1+20)), font, fontScale, (255,255,255), thickness ,cv2.LINE_AA)

        # Publish visualization image
        image_msg = self.bridge.cv2_to_imgmsg(imgOut, "rgb8")
        self.published_image_topic.publish(image_msg)
        

def main(args=None):
    # Initialize node
    rclpy.init(args=args)
    try:
        # Define detector object
        dm = DetectorManager()
        try:
            rclpy.spin(dm)
        except KeyboardInterrupt:
            dm.get_logger().info('Keyboard Interrupt (SIGNT)')
        finally:
            dm.destroy_node()
    finally:
        rclpy.shutdown()

if __name__=="__main__":
    main()