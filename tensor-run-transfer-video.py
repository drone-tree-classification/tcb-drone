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
    parser = argparse.ArgumentParser(description='Run Grid-Based ResNet Tree Detector on an video')
    parser.add_argument('video', help='Path to the video')
    parser.add_argument('-m', '--model', default='FineTunedResNetOD.keras', help='Path to the trained model')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Confidence threshold for bounding boxes')
    parser.add_argument('-o', '--output', help='Optional path to save the output image')
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: Image {args.image} not found.")
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"Error: Model {args.model} not found.")
        sys.exit(1)

    print(f"Loading model {args.model}...")
    model = tf.keras.models.load_model(args.model, compile=False)

    IMG_SIZE = 512
    GRID_SIZE = 16

    cap = cv2.VideoCapture(args.video)
    while cap.isOpened():

        ret, im = cap.read()

        if not ret:

            break  # End of video

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

        for box in boxes:

            # label = f"{names[int(cls_idx)]}: {score:.2f}"

            # Draw bounding box rectangle

            cv2.rectangle(im_resized, (box.xmin, box.ymin), (box.xmax, box.ymax), (0, 255, 0), 2)

            # Draw label background and text string

            # cv2.putText(frame, label, (xmin, ymin - 10),

                                    # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # 5. Display the processed frame

        cv2.imshow('Pre-trained Object Detection', im_resized)

        # Break out of loop if 'q' key is pressed

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break

    cap.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

