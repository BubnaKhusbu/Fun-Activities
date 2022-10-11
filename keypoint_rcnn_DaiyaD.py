#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:48:50 2021

@author: khusbububna
"""

import torch
import torchvision
import cv2
import argparse
import utils
import time
from PIL import Image
from torchvision.transforms import transforms as transforms
import matplotlib.pyplot as plt
import urllib
#parser = argparse.ArgumentParser()
#parser.add_argument('-i', '--input', required=True, 
 #                   help='path to the input data')
#args = vars(parser.parse_args())
# transform to convert the image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,
                                                               num_keypoints=17)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

cap = cv2.VideoCapture('./input/WinniePoohExpedition.mp4')
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')
# get the video frames' width and height
frame_width = 640
frame_height = 360

save_path = './input/output_video_WinnieExpedition.mp4'
# define codec and create VideoWriter object 
out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, 
                      (frame_width, frame_height))
out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('m','p','4','v'), 5, 
                      (frame_width, frame_height))
#out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M','J','P','G'), 5, 
#                      (frame_width, frame_height))
frame_count = 256 
total_fps = 0 

while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        
        pil_image = Image.fromarray(frame).convert('RGB')
        orig_frame = frame
        # transform the image
        image = transform(pil_image)
        # add a batch dimension
        image = image.unsqueeze(0).to(device)
        # get the start time
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image)
        # get the end time
        end_time = time.time()
        output_image = utils.draw_keypoints(outputs, orig_frame)
        # get the fps
        fps = 1 / (end_time - start_time)
        total_fps += fps
        frame_count += 1
        wait_time = max(1, int(fps/4))
        #cv2.imshow('Pose detection frame', output_image)
        #out.write(output_image)
        img=plt.imshow(output_image)
        plt.show()
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
cv2.destroyAllWindows()
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
out.release()