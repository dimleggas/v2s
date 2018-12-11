#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the functions we used to preprocess audios from the videos
and store them as numpy arrays and a function to generate audio files in wav
format from the audio arrays generated by our model.
"""

import os
import csv
import numpy as np
import moviepy.editor as mp
from scipy.io import wavfile
from keras.preprocessing.sequence import pad_sequences


'''
Scans a directory and returns a sorted list of filepaths to mp4 files in the 
directory
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
Loads a video, extracts the audio and returns an audio dictionary containing
frame count, frame rate and audio frames represented as a numpy array
'''
def load_audio(filepath):
    
    # Load video
    vid_clip = mp.VideoFileClip(filepath)
    
    # Extract audio frames as a numpy array
    audio_clip = vid_clip.audio.to_soundarray()
    
    audio = {"frame_count": audio_clip.shape[0],
             "frame_rate": 44100,
             "frames": audio_clip}
    
    vid_clip.close()
    
    # Return audio dictionary
    return audio

'''
Pads audios to 10 seconds using same post padding and post truncating
'''
def pad_audio(audio):
        
    frame_rate = audio["frame_rate"]
    frames = audio["frames"]
    
    # Determine the frame count equivalent of 10 seconds
    maxlen = int(10 * frame_rate)
    
    # Same post padding and post truncating
    padded_frames = pad_sequences([frames], dtype=np.ndarray, maxlen=maxlen, 
                                  padding='post', truncating='post',
                                  value=frames[-1])[0]
    
    audio["frames"] = padded_frames
    audio["frame_count"] = len(padded_frames)
    
    # Return audio dictionary with padded frames
    return audio

'''
Samples video frames at a rate of approximately 16kHz
'''
def sample_audio(audio, n=159744):
    
    frames = audio["frames"]
    length = float(audio["frame_count"])
    sample_idx = np.linspace(0, length, num=n, endpoint=False, dtype=int)
    audio["frames"] = frames[sample_idx]
    
    # Returns the audio dictionary with 159744 audio frames (1024 audio 
    # samples per video frame)
    return audio

'''
Given input and output directory paths, preprocesses all mp4 files in the 
input directory and saves (2, 159744) audio numpy arrays in the output 
directory. Even though we trained our models with mono audio, we stored them as
stereo in case we wanted to observe stereo performace.
'''
def videos2audio(dir_from, dir_to):
    
    # Scan the input directory and get the filepaths to the mp4 files
    video_filepaths = scan_directory(dir_from)
    total = len(video_filepaths)
    
    for filepath in video_filepaths:
        
        # Preprocess the audio
        audio = load_audio(filepath)
        audio = pad_audio(audio)
        audio = sample_audio(audio)
        
        # Store the audio features in the output directory
        filename = filepath.split('/')[-1].split('.')[-2]
        out_path = dir_to + filename
        np.savez_compressed(out_path, features=audio["frames"].transpose())

    return True

'''
Given the filepath of the generated audios in array form, stores them as wav
format audio files
'''
def generate_audio(csv_path, out_dir):
    
    # Read generated audio arrays from a csv file
    audios = []
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # Convert to float
        for row in csv_reader:
            audios.append([float(x) for x in row])

    # Convert to numpy arrays and store wav format audio files
    r = np.zeros((len(audios),len(audios[5])))
    for i in range(len(audios)):
        r[i] = np.array(audios[i])
    for i in range(r.shape[0]):
        wavfile.write(out_dir+str(i)+'.wav', 16000, r[i])
    
    return True
