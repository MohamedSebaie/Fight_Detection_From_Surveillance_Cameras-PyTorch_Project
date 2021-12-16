import torch
import torchvision
import cv2
from google.colab.patches import cv2_imshow
import argparse
import time
import numpy as np
import albumentations as A

# Define the transforms
def transform_():
	transform = A.Compose(
    [A.Resize(128, 171, always_apply=True),A.CenterCrop(112, 112, always_apply=True),
     A.Normalize(mean = [0.43216, 0.394666, 0.37645],std = [0.22803, 0.22145, 0.216989], always_apply=True)]
     )
	return transform


def frames_extraction(video_path,SEQUENCE_LENGTH):
    '''
    This function will extract the required frames from a video after resizing and normalizing them.
    Args:
        video_path: The path of the video in the disk, whose frames are to be extracted.
        SEQUENCE_LENGTH: TThe number of Frames we want.
    Returns:
        frames_list: A list containing the resized and normalized frames of the video.
    '''

    # Declare a list to store video frames.
    frames_list = []
    
    # Read the Video File using the VideoCapture object.
    video_reader = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

    transform= transform_()

    # Iterate through the Video Frames.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Reading the frame from the video. 
        success, frame = video_reader.read() 

        # Check if Video frame is not successfully read then break the loop
        if not success:
            break

        image = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(image=frame)['image']
        
        # Append the normalized frame into the frames list
        frames_list.append(frame)
    
    # Release the VideoCapture object. 
    video_reader.release()

    # Return the frames list.
    return frames_list