
 

<h1 color="green"><b>Fight Detection Package</b></h1>

---

<h1 color="green"><b>Abstract</b></h1>
<p>Human action recognition can be seen as the automatic labeling of a video according to the actions occurring in it. It has
become one of the most challenging and attractive problems in the pattern recognition and video classification fields.
The problem itself is difficult to solve by traditional video processing methods because of several challenges such as
the background noise, sizes of subjects in different videos, and the speed of actions.Derived from the progress of
deep learning methods, several directions are developed to recognize a human action from a video, such as the
long-short-term memory (LSTM)-based model, two-stream convolutional neural network (CNN) model, and the convolutional 3D model.
Human action recognition is used in some surveillance systems and video processing tools.
Our main problem is Fight Detection which we achieved to solve by using transfer learning on pretrained convolutional 3D models
that aim to recognize the motions and actions of humans.
All models use Kinetics-400 dataset for the pretrained part and Vision-based Fight Detection From Surveillance Cameras dataset
for the finetuned part.</p>

<h1 color="green"><b>Results</b></h1>
<table style="width:100%">
  <tr>
    <th>Model</th>
    <th>Top-1 Accuracy</th>
    <th>Batch Size (Videos)</th>
    <th>Input Frames</th>
    <th>Inference Time (Videos/sec)</th>
  </tr>
  
  <tr>
    <td>r2plus1d_18</td> <td>82.22%</td>  <td>4</td>  <td>16</td>  <td>11.3</td>
  </tr>
 
 <tr>
    <td>r3d_18</td> <td>88.89%</td>  <td>4</td>  <td>16</td>  <td>11.3</td>
  </tr>
 
 <tr>
    <td>mc3_18</td> <td>91.11%</td>  <td>4</td>  <td>16</td>  <td>11.3</td>
  </tr>
 
 <tr>
    <td>mc3_18</td> <td>91.11%</td>  <td>8</td>  <td>16</td>  <td>11.3</td>
  </tr>
 
 <tr>
    <td>mc3_18</td> <td>83.72%</td>  <td>4</td>  <td>32</td>  <td>5.63</td>
  </tr>
  
</table>



<h1 color="green"><b>Pytorch Pretrained Models</b></h1>
<p>All pretrained models can be found in this link.
 <a href="https://pytorch.org/vision/stable/models.html">lhttps://pytorch.org/vision/stable/models.html</a></p>


<h1 color="green"><b>Instructions to Install our Fight Detection Package</b></h1>


1. Install:

```python
pip install Fight-Detection
pip install pytube
```

2. Detect Fight or Not by Pass your Local Video:

```python
from fight_detection import Fight_utils
# Run the Below Function by Input your Test Video Path to get the outPut Video with Fight Detection or Not
Fight_utils.fightDetection(inputPath,seq,skip,outputPath,showInfo)
```
3. Show the Output Video with Detection:

```python
from moviepy.editor import *
VideoFileClip(outputPath, audio=False, target_resolution=(300,None)).ipython_display()
```
4. To Start Detect the Fight on Streaming

```python
Fight_utils.start_streaming(streamingURL)
```