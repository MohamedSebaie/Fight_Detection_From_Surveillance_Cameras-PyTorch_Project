import torch
from FDSC.utils.Fight_utils import loadModel, predict_on_video,start_streaming
import argparse
import time


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch STAM Kinetics Inference')
parser.add_argument('--modelPath')
parser.add_argument('--streaming', type=bool, default=False)
parser.add_argument('--inputPath')
parser.add_argument('--outputPath')
parser.add_argument('--sequenceLength', type=int, default=16)
parser.add_argument('--skip', type=int, default=2)
parser.add_argument('--showInfo', type=bool, default=False)




def main():
    # parsing args
    args = parser.parse_args()


    model = loadModel(args.modelPath)
    # Perform Fight Detection on the Test Video.
    if args.streaming==True:
        start_streaming(model,args.inputPath)

    else:
        start=time.time()
        predict_on_video(args.inputPath, args.outputPath, model, args.sequenceLength, args.skip, args.showInfo) 
        end = time.time()
        print(end-start)


if __name__ == '__main__':
    main()