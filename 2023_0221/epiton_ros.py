import os
import time
import math
import numpy as np
import copy
import cv2
import argparse

from detection.det_infer import Predictor
from segmentation.seg_infer import BaseEngine

import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseArray, Pose

###
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
###

class Camemra_Node:
    def __init__(self,args):
        rospy.init_node('Camemra_node')
        self.args = args

        mode =['save','full','match']
        self.mode = mode[1]
        model_sel =['all','det','seg']
        self.model_sel = model_sel[0]
        self.img_flag = [1,0,0]
        self.divid =False

        ### Seg Model initiolization
        seg_model = args.seg_weight
        seg_input_size = (384, 640)
        anchors = args.anchor if args.anchor else None
        nc = int(args.nc)
        self.seg_pred = BaseEngine(seg_model, seg_input_size, anchors, nc)

        ### Det Model initiolization
        self.det_pred = Predictor(engine_path=args.det_weight)
        self.det_pred.get_fps()

        ###Inintilize flag and box
        self.get_f60_new_image = False
        self.Camera_f60_bbox = None
        self.bbox_f60 = PoseArray()

        self.get_f120_new_image = False
        self.Camera_f120_bbox = None
        self.bbox_f120 = PoseArray()

        self.get_r120_new_image = False
        self.Camera_r120_bbox = None
        self.bbox_r120 = PoseArray()

        self.img_flag = [1,0,0]

        self.cur_f60_img = {'img':None, 'header':None}
        self.sub_f60_img = {'img':None, 'header':None}
        
        self.cur_f120_img = {'img':None, 'header':None}
        self.sub_f120_img = {'img':None, 'header':None}
        
        self.cur_r120_img = {'img':None, 'header':None}
        self.sub_r120_img = {'img':None, 'header':None}

        self.pub_od_f60 = rospy.Publisher('/camera_od/front_60', PoseArray, queue_size=1)
        self.pub_od_f120 = rospy.Publisher('/camera_od/front_120', PoseArray, queue_size=1)
        self.pub_od_r120 = rospy.Publisher('/camera_od/rear_120', PoseArray, queue_size=1)
        
        rospy.Subscriber('/gmsl_camera/dev/video0/compressed', CompressedImage, self.IMG_f60_callback)
        rospy.Subscriber('/gmsl_camera/dev/video1/compressed', CompressedImage, self.IMG_f120_callback)
        rospy.Subscriber('/gmsl_camera/dev/video2/compressed', CompressedImage, self.IMG_r120_callback)

       ##########################
        self.pub_f60_det = rospy.Publisher('/det_result/f60', Image, queue_size=1)
        self.pub_f60_seg = rospy.Publisher('/seg_result/f60', Image, queue_size=1)
        self.pub_f120_det = rospy.Publisher('/det_result/f120', Image, queue_size=1)
        self.pub_f120_seg = rospy.Publisher('/seg_result/f120', Image, queue_size=1)
        self.pub_r120_det = rospy.Publisher('/det_result/r120', Image, queue_size=1)
        self.pub_r120_seg = rospy.Publisher('/seg_result/r120', Image, queue_size=1)
        self.bridge = CvBridge()
        self.is_save =False
        self.sup = []
        ##########################
        
    def IMG_f60_callback(self,msg):
        if not self.get_f60_new_image:
            np_arr = np.fromstring(msg.data, np.uint8)
            front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # front_img = cv2.resize(front_img, (self.img_shape))
            self.cur_f60_img['img'] = front_img
            self.cur_f60_img['header'] = msg.header
            self.get_f60_new_image = True

    def IMG_f120_callback(self,msg):
        if not self.get_f120_new_image:
            np_arr = np.fromstring(msg.data, np.uint8)
            front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # front_img = cv2.resize(front_img, (self.img_shape))
            
            self.cur_f120_img['img'] = front_img
            self.cur_f120_img['header'] = msg.header
            self.get_f120_new_image = True

    def IMG_r120_callback(self,msg):
        if not self.get_r120_new_image:
            np_arr = np.fromstring(msg.data, np.uint8)
            front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # front_img = cv2.resize(front_img, (self.img_shape))
            
            self.cur_r120_img['img'] = front_img
            self.cur_r120_img['header'] = msg.header
            self.get_r120_new_image = True

    def pose_set(self,bboxes,flag):
        for bbox in bboxes:
            pose = Pose()
            pose.position.x = bbox[0]# box class
            pose.position.y = bbox[1]# box area
            pose.position.z = bbox[2]# box score
            pose.orientation.x = bbox[3][0]# box mid x
            pose.orientation.y = bbox[3][1]# box mid y

            if flag == 'f60':
                self.bbox_f60.poses.append(pose)
            if flag == 'f120':
                    self.bbox_f120.poses.append(pose)
            if flag == 'r120':
                    self.bbox_r120.poses.append(pose)

    def det_pubulissher(self,det_img,det_box,flag):
        temp = time.time()
        if flag == 'f60':
            det_f60_msg = self.bridge.cv2_to_imgmsg(det_img, "bgr8")#color
            self.pose_set(det_box,flag)
            self.pub_od_f60.publish(self.bbox_f60)
            self.pub_f60_det.publish(det_f60_msg)
            print('f60 publishing is :', round((time.time() - temp)*1000,2),' ms')
            print('f60 whole processing is :', round((time.time() - self.t1)*1000,2),' ms')

        if flag == 'f120':
            det_f120_msg = self.bridge.cv2_to_imgmsg(det_img, "bgr8")#color
            self.pose_set(det_box,flag)
            self.pub_od_f120.publish(self.bbox_f120)
            self.pub_f120_det.publish(det_f120_msg)
            print('f120 publishing is :', round((time.time() - temp)*1000,2),' ms')
            print('f120 whole processing is :', round((time.time() - self.t2)*1000,2),' ms')

        if flag == 'r120':
            det_r120_msg = self.bridge.cv2_to_imgmsg(det_img, "bgr8")#color
            self.pose_set(det_box,flag)
            self.pub_od_r120.publish(self.bbox_r120)
            self.pub_r120_det.publish(det_r120_msg)
            print('r120 publishing is :', round((time.time() - temp)*1000,2),' ms')
            print('r120 whole processing is :', round((time.time() - self.t3)*1000,2),' ms')

    def seg_pubulissher(self,seg_img,flag):
        temp = time.time()
        if flag == 'f60':
            seg_f60_msg = self.bridge.cv2_to_imgmsg(np.array(seg_img), "bgr8")#gray
            self.pub_f60_seg.publish(seg_f60_msg)
            print('f60 publishing is :', round((time.time() - temp)*1000,2),' ms')
            print('f60 whole processing is :', round((time.time() - self.t1)*1000,2),' ms')

        if flag == 'f120':
            seg_f120_msg = self.bridge.cv2_to_imgmsg(np.array(seg_img), "bgr8")#gray
            self.pub_f120_seg.publish(seg_f120_msg)
            print('f120 publishing is :', round((time.time() - temp)*1000,2),' ms')
            print('f120 whole processing is :', round((time.time() - self.t2)*1000,2),' ms')

        if flag == 'r120':
            seg_r120_msg = self.bridge.cv2_to_imgmsg(np.array(seg_img), "bgr8")#gray
            self.pub_r120_seg.publish(seg_r120_msg)
            print('r120 publishing is :', round((time.time() - temp)*1000,2),' ms')
            print('r120 whole processing is :', round((time.time() - self.t3)*1000,2),' ms')

    def pubulissher(self,seg_img,det_img,det_box,flag):
        ####    
        temp = time.time()
        if flag == 'f60':
            det_f60_msg = self.bridge.cv2_to_imgmsg(det_img, "bgr8")#color
            seg_f60_msg = self.bridge.cv2_to_imgmsg(np.array(seg_img), "bgr8")#gray
            self.pose_set(det_box,flag)
            self.pub_od_f60.publish(self.bbox_f60)
            self.pub_f60_det.publish(det_f60_msg)
            self.pub_f60_seg.publish(seg_f60_msg)
            print('f60 publishing is :', round((time.time() - temp)*1000,2),' ms')
            print('f60 whole processing is :', round((time.time() - self.t1)*1000,2),' ms')

        if flag == 'f120':
            det_f120_msg = self.bridge.cv2_to_imgmsg(det_img, "bgr8")#color
            seg_f120_msg = self.bridge.cv2_to_imgmsg(np.array(seg_img), "bgr8")#gray
            self.pose_set(det_box,flag)
            self.pub_od_f120.publish(self.bbox_f120)
            self.pub_f120_det.publish(det_f120_msg)
            self.pub_f120_seg.publish(seg_f120_msg)
            print('f120 publishing is :', round((time.time() - temp)*1000,2),' ms')
            print('f120 whole processing is :', round((time.time() - self.t2)*1000,2),' ms')

        if flag == 'r120':
            det_r120_msg = self.bridge.cv2_to_imgmsg(det_img, "bgr8")#color
            seg_r120_msg = self.bridge.cv2_to_imgmsg(np.array(seg_img), "bgr8")#gray
            self.pose_set(det_box,flag)
            self.pub_od_r120.publish(self.bbox_r120)
            self.pub_r120_det.publish(det_r120_msg)
            self.pub_r120_seg.publish(seg_r120_msg)
            print('r120 publishing is :', round((time.time() - temp)*1000,2),' ms')
            print('r120 whole processing is :', round((time.time() - self.t3)*1000,2),' ms')

    def image_process(self,img,flag):
        try:
            t2= time.time()
            if self.model_sel == 'all' and not self.mode == 'match':
                if not self.divid:
                    det_img, box_result = self.det_pred.steam_inference(img,conf=0.1, end2end=args.end2end)
                    _,_,_,_ = self.seg_pred.inference(img)
                    seg_img, t_ms = self.seg_pred.draw_2D(img)
                    self.pubulissher(seg_img,det_img,box_result,flag)

                else:
                    if flag == 'f60':
                        det_img, box_result = self.det_pred.steam_inference(img,conf=0.1, end2end=args.end2end)
                        self.det_pubulissher(det_img,box_result,flag)
        
                    if flag == 'f120':
                        _,_,_,_ = self.seg_pred.inference(img)
                        seg_img, t_ms = self.seg_pred.draw_2D(img)
                        self.seg_pubulissher(seg_img,flag)
                

            elif self.model_sel == 'det' and not self.mode == 'match':
                det_img, box_result = self.det_pred.steam_inference(img,conf=0.1, end2end=args.end2end)
                seg_img = None
                self.det_pubulissher(det_img,box_result,flag)

            elif self.model_sel == 'seg' and not self.mode == 'match':
                det_img, box_result = None, None
                _,_,_,_ = self.seg_pred.inference(img)
                seg_img, t_ms = self.seg_pred.draw_2D(img)
                self.seg_pubulissher(seg_img,flag)


            elif self.mode == 'match':
                if flag == 'f60' :
                    det_img, box_result = self.det_pred.steam_inference(img,conf=0.1, end2end=args.end2end)
                    self.det_pubulissher(det_img,box_result,flag)
    
                if flag == 'f120' :
                    _,_,_,_ = self.seg_pred.inference(img)
                    seg_img, t_ms = self.seg_pred.draw_2D(img)
                    self.seg_pubulissher(seg_img,flag)
                
                if flag == 'r120' :
                    _,_,_,_ = self.seg_pred.inference(img)
                    seg_img, t_ms = self.seg_pred.draw_2D(img)
                    self.seg_pubulissher(seg_img,flag)

        except CvBridgeError as e:
            print(e)

    def main(self):
        while not rospy.is_shutdown():
            if self.mode == 'save':
                self.t1 = time.time()
                if self.img_flag[0] == 1 and self.get_f60_new_image:
                    self.sub_f60_img['img'] = self.cur_f60_img['img']
                    orig_im_f60 = copy.copy(self.sub_f60_img['img']) 
                    self.image_process(orig_im_f60,'f60')
                    self.get_f60_new_image = False
                    self.img_flag = [0,1,0]

                self.t2 = time.time()
                if self.img_flag[1] == 1 and self.get_f120_new_image:
                    self.sub_f120_img['img'] = self.cur_f120_img['img']
                    orig_im_f120 = copy.copy(self.sub_f120_img['img']) 
                    self.image_process(orig_im_f120,'f120')
                    self.get_f120_new_image = False
                    self.img_flag = [0,0,1]

                self.t3 = time.time()
                if self.img_flag[2] == 1 and self.get_r120_new_image:
                    self.sub_r120_img['img'] = self.cur_r120_img['img']
                    orig_im_r120 = copy.copy(self.sub_r120_img['img']) 
                    self.image_process(orig_im_r120,'r120')
                    self.get_r120_new_image = False
                    self.img_flag = [1,0,0]
                
            elif self.mode == 'full' or self.mode == 'match':
                self.t1 = time.time()
                if self.get_f60_new_image and self.img_flag[0] == 1 :
                    print('f60 ------')
                    self.sub_f60_img['img'] = self.cur_f60_img['img']
                    orig_im_f60 = copy.copy(self.sub_f60_img['img']) 
                    self.image_process(orig_im_f60,'f60')
                    self.get_f60_new_image = False
                    self.img_flag = [0,1,0]

                self.t2 = time.time()
                if self.get_f120_new_image and self.img_flag[1] == 1:
                    print('f120 =====')
                    self.sub_f120_img['img'] = self.cur_f120_img['img']
                    orig_im_f120 = copy.copy(self.sub_f120_img['img']) 
                    self.image_process(orig_im_f120,'f120')
                    self.get_f120_new_image = False
                    self.img_flag = [0,0,1]

                self.t3 = time.time()
                if self.get_r120_new_image and self.img_flag[2] == 1:
                    print('r120 !!!!!!')
                    self.sub_r120_img['img'] = self.cur_r120_img['img']
                    orig_im_r120 = copy.copy(self.sub_r120_img['img']) 
                    self.image_process(orig_im_r120,'r120')
                    self.get_r120_new_image = False
                    self.img_flag = [1,0,0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--det_weight', default="./detection/weights/epitone_7x.trt")  
    parser.add_argument('--seg_weight', default="/home/cvlab-swlee/Desktop/master/git/epiton/ros_vision/segmentation/weights/hybridnets_c0_384x640_simplified.trt")  

    parser.add_argument("--end2end", default=False, action="store_true",help="use end2end engine")
    
    parser.add_argument('--anchor', type=str, required=True, help='The anchors numpy file path')
    parser.add_argument('--nc', type=str, default='10', help='Number of detection classes')
    
    args = parser.parse_args()

    Camemra_Node = Camemra_Node(args)
    Camemra_Node.main()

