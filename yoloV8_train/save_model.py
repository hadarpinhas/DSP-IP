import numpy as np
from time import time
from slice import slice_img_with_overlap
from PIL import Image
from sys import argv
import argparse

def save_model(modelweights,savepath):
    from yolo_v8_test import save_model_onnx
    save_model_onnx(modelweights,savepath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",required=True, help="Model weights")
    parser.add_argument("--savepath",required=True, help="export save path")
    args = parser.parse_args()
    if args.model :
        save_model(args.model,args.savepath)

