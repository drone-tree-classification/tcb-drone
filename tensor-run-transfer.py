#!./tensorflow/bin/python

import argparse
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run Grid-Based ResNet Tree Detector on an image')
    parser.add_argument('image', help='Path to the input image')
    parser.add_argument('-m', '--model', default='FineTunedResNetOD.keras', help='Path to the trained model')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Confidence threshold for bounding boxes')
    parser.add_argument('-o', '--output', help='Optional path to save the output image')
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: Image {args.image} not found.")
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"Error: Model {args.model} not found.")
        sys.exit(1)

    print(f"Loading model {args.model}...")
    model = tf.keras.models.load_model(args.model, compile=False)

    IMG_SIZE = 512
    GRID_SIZE = 16

    im = cv2.imread(args.image)
    if im is None:
        print(f"Error: Could not read image {args.image}")
        sys.exit(1)

    orig_h, orig_w = im.shape[:2]

    im_resized = cv2.resize(im, (IMG_SIZE, IMG_SIZE))
    im_processed = preprocess_input(im_resized.astype(np.float32))

    im_batch = np.expand_dims(im_processed, axis=0)

    print("Running inference...")
    pred = model.predict(im_batch, verbose=0)[0] 

    boxes = []
    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            conf = pred[gy, gx, 0]
            if conf >= args.threshold:
                dx = pred[gy, gx, 1]
                dy = pred[gy, gx, 2]
                norm_w = pred[gy, gx, 3]
                norm_h = pred[gy, gx, 4]

                cx = ((gx + dx) / float(GRID_SIZE)) * orig_w
                cy = ((gy + dy) / float(GRID_SIZE)) * orig_h
                w = norm_w * orig_w
                h = norm_h * orig_h

                xmin = cx - w / 2.0
                ymin = cy - h / 2.0

                boxes.append({
                    'xmin': xmin,
                    'ymin': ymin,
                    'w': w,
                    'h': h,
                    'conf': conf
                })

    print(f"Found {len(boxes)} bounding boxes with confidence >= {args.threshold}")

    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(im_rgb)

    for b in boxes:
        rect = patches.Rectangle((b['xmin'], b['ymin']), b['w'], b['h'], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(b['xmin'], b['ymin'] - 5, f"{b['conf']:.2f}", color='red', fontsize=12, weight='bold')

    plt.axis('off')
    plt.title(f"Detections (threshold={args.threshold})")
    
    if args.output:
        plt.savefig(args.output, bbox_inches='tight')
        print(f"Saved output to {args.output}")
    else:
        plt.show()

if __name__ == '__main__':
    main()

