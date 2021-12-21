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

<h1 style="color: blue"><b>Confusion Matrix</b></h1>
<img src="/html/images/test.png" alt="Simply Easy Learning" width="200" height="80">



<h1 color="green"><b>Finetuned model Weights</b></h1>
<p>mc3_18-->Link(model path)</p>



<h1 color="green"><b>Instructions</b></h1>








<h1 color="green"><b>Inference</b></h1>
<p>Run the infer.py script and pass the required arguments (model path, input & output paths, sequence length, skip frames) <br>
python -m infer \ <br>
--modelPath="/to/model_16_70_4_0.88.pth" \ <br>
--inputPath="/to/input.mp4" \ <br>
--outputPath="/to/output.mp4" \ <br>
--sequenceLength=16 \ <br>
--skip=2 \ <br>
--showInfo=True</p> <br>



