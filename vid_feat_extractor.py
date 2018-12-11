#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the functions we used to preprocess videos and extract video
feature vectors (4096D) for each frame using the first fully-connected layer of
VGG19, a very deep convolutional network for image classification.
"""


import os
import numpy as np
import cv2
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.models import Model


'''
Loads image classifier, VGG19 with the weights trained on the ImageNet dataset. 
Returns a model that outputs the feature representations after the first 
fully-connected layer of the network.
'''
def load_VGG19():
    
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    
    return model

'''
Given the filepath, loads a video, resizes the frames to the size VGG19 was
trained on and returns a dictionary containing frame count, frame rate and the
list of frames in the video.
'''
def load_video(filepath):
   
    # Load the video
    reader = cv2.VideoCapture(filepath)
    p = True
    frames = []
    
    # Record the frame rate and frame count for pre-processing
    video = {"frame_count": reader.get(cv2.CAP_PROP_FRAME_COUNT),
             "frame_rate": reader.get(cv2.CAP_PROP_FPS)}
    
    # Iterate over the video, resizing the frames to (224x224)
    while p:
        data = reader.read()
        p = data[0]
        if p:
            image = data[1]
            image = cv2.resize(image, (224,224))
            frames.append(image)
    
    reader.release()
    video["frames"] = frames
    
    # Return video dictionary
    return video

'''
Pads videos to 10 seconds using same post padding and post truncating.
'''
def pad_video(video):
    
    fr_rate = video["frame_rate"]
    fr_count = video["frame_count"]
    frames = video["frames"]
    
    # Determine the frame count equivalent of 10 seconds
    maxlen = int(10 * fr_rate)
    padded_frames = frames
    
    # Apply same post padding and post truncating to make all videos 10 seconds
    # while keeping the frame rate constant
    padded_frames = pad_sequences([frames], dtype=object, maxlen=maxlen, 
                                  padding='post', truncating='post',
                                  value=frames[-1])[0]
    
    video["frames"] = padded_frames
    video["frame_count"] = len(padded_frames)
    
    # Return video dictionary with padded frames
    return video
    
'''
Samples video frames at a rate of 15.6 frames per second.
'''
def sample_video(video, n):
    
    frames = video["frames"]
    length = float(video["frame_count"])    
    sample_idx = np.linspace(0, length, num=n, endpoint=False, dtype=int)
    video["frames"] = frames[sample_idx]
    
    # Returns video dictionary with 15.6 * 10 = 156 frames
    return video
    
'''
Scans a given directory for mp4 files and returns a sorted list of filepaths
'''
def scan_directory(dir_path):
    
    video_filenames = []
    for path, dirs, files in os.walk(dir_path):
        for f in files:
            abs_path = path + f
            ext = f.split('.')[-1]
            if ext == 'mp4':
                video_filenames.append(abs_path)
    
    video_filenames.sort()
    return video_filenames

'''
Given input and output directory paths, preprocesses all mp4 files in the 
input directory and saves (156, 4096) video features for each video in the
output directory.
'''
def videos2feat(dir_from, dir_to):
   
    # Load the video feature extraction model
    model = load_VGG19()
    
    # Scan the input directory for mp4 filepaths
    video_filepaths = scan_directory(dir_from)
    total = len(video_filepaths)
    
    for filepath in video_filepaths:
        
        # Load and preprocess video frames
        video = load_video(filepath)
        video = pad_video(video)
        video = sample_video(video, 156)
        frames = preprocess_input(video["frames"])        
        
        # Extract video features
        feats = model.predict(frames)
        
        # Store video features in the output directory
        filename = filepath.split('/')[-1].split('.')[-2]
        out_path = dir_to + filename
        np.savez_compressed(out_path, features=feats)
    
    return True
    
    
    
    
    