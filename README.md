# Fight_Detection_From_Surveillance_Cameras-PyTorch_Project
 
TO Test the Project -----> Clone the Repo then run the infer.py script. 
For example, for mc3_18
run:
```bash
python -m infer 
--modelPath=".\FDSC\models\model_16_m3_0.8888.pth" 
--inputPath=".\testVideo.mp4" 
--outputPath=".\outVideo.mp4" 
--sequenceLength=16 
--skip=2 
--showInfo=True
```

<h1 style="color="rblue"><b>Abstract</b></h1>
<p>Human action recognition can be seen as the automatic labeling of a video according to the actions occurring in it. It has
become one of the most challenging and attractive problems in the pattern recognition and video classification fields.
The problem itself is difficult to solve by traditional video processing methods because of several challenges such as
the background noise, sizes of subjects in different videos, and the speed of actions.Derived from the progress of
deep learning methods, several directions are developed to recognize a human action from a video, such as the
long-short-term memory (LSTM)-based model, two-stream convolutional neural network (CNN) model, and the convolutional 3D model.
Human action recognition is used in some surveillance systems and video processing tools.
Our main problem is Fight Detection which we achieved to solve by using transfer learning on pretrained convolutional 3D models
that aim to recognize the motions and actions of humans.
All models use Kinetics-400 dataset for the pretrained part and Vision-based Fight Detection From Surveillance Cameras dataset</p>
for the finetuned part.

