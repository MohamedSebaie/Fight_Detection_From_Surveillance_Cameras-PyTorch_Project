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
