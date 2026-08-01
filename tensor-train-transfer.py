#!./tensorflow/bin/python

import os
import sys
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

PROGRAM_NAME = Path(sys.argv[0]).name

# Configure GPU dynamic memory allocation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{PROGRAM_NAME}: Info: Found {len(gpus)} GPU(s). Enabled dynamic memory growth.", file=sys.stderr)
    except RuntimeError as e:
        print(f"{PROGRAM_NAME}: Warning: GPU configuration failed: {e}", file=sys.stderr)
else:
    print(f"{PROGRAM_NAME}: Info: No GPU detected. Training will run on CPU.", file=sys.stderr)


def parse_pascal_voc_xml(path):
    """Parses Pascal VOC XML annotation file and extracts bounding boxes."""
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return False, 0, 0, []

    width, height = 0, 0
    size_tag = root.find('size')
    if size_tag is not None:
        try:
            width = int(size_tag.find('width').text)
            height = int(size_tag.find('height').text)
        except (AttributeError, ValueError):
            pass

    boxes = []
    for obj_tag in root.findall('object'):
        bndbox = obj_tag.find('bndbox')
        if bndbox is not None:
            try:
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                boxes.append({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
            except (AttributeError, ValueError):
                continue

    return True, width, height, boxes


def load_dataset_pairs(dataset_dirs):
    """Finds matching JPG and XML pairs across directory paths."""
    xml_files, jpg_files = [], []
    for ds_dir in dataset_dirs:
        for root, _, files in os.walk(ds_dir):
            for file in files:
                ext = Path(file).suffix.lower()
                full_path = os.path.join(root, file)
                if ext == ".xml":
                    xml_files.append(full_path)
                elif ext in [".jpg", ".jpeg"]:
                    jpg_files.append(full_path)

    jpg_dict = {str(Path(f).with_suffix('')): f for f in jpg_files}
    matched_pairs = []
    for xml_path in xml_files:
        key = str(Path(xml_path).with_suffix(''))
        if key in jpg_dict:
            matched_pairs.append({'image': jpg_dict[key], 'xml': xml_path})

    return matched_pairs


def make_dataset_generator(matched_pairs, img_size, grid_size):
    """Generator yielding processed images and target grids on-the-fly."""
    for item in matched_pairs:
        success, orig_w, orig_h, boxes = parse_pascal_voc_xml(item['xml'])
        if not success or not boxes:
            continue

        im = cv2.imread(item['image'])
        if im is None:
            continue

        if orig_w == 0 or orig_h == 0:
            orig_h, orig_w, _ = im.shape

        # Preprocess image for ResNet50 (Keep BGR, apply ImageNet mean subtraction)
        im_resized = cv2.resize(im, (img_size, img_size))
        im_processed = preprocess_input(im_resized.astype(np.float32))

        # Initialize Target Grid: (GRID_SIZE, GRID_SIZE, 5) -> [conf, dx, dy, norm_w, norm_h]
        grid_target = np.zeros((grid_size, grid_size, 5), dtype=np.float32)

        scale_x = img_size / float(orig_w)
        scale_y = img_size / float(orig_h)

        for b in boxes:
            xmin = np.clip(b['xmin'] * scale_x, 0, img_size)
            xmax = np.clip(b['xmax'] * scale_x, 0, img_size)
            ymin = np.clip(b['ymin'] * scale_y, 0, img_size)
            ymax = np.clip(b['ymax'] * scale_y, 0, img_size)

            w = xmax - xmin
            h = ymax - ymin
            cx = xmin + (w / 2.0)
            cy = ymin + (h / 2.0)

            if w <= 0 or h <= 0:
                continue

            # Absolute grid coordinates
            grid_x_float = (cx / img_size) * grid_size
            grid_y_float = (cy / img_size) * grid_size

            grid_x = int(np.clip(grid_x_float, 0, grid_size - 1))
            grid_y = int(np.clip(grid_y_float, 0, grid_size - 1))

            # Local offsets relative to grid cell [0, 1]
            dx = grid_x_float - grid_x
            dy = grid_y_float - grid_y

            # Dimensions normalized relative to whole image [0, 1]
            norm_w = w / img_size
            norm_h = h / img_size

            # Mark cell target
            grid_target[grid_y, grid_x, 0] = 1.0
            grid_target[grid_y, grid_x, 1:5] = [dx, dy, norm_w, norm_h]

        yield im_processed, grid_target


def weighted_od_loss(y_true, y_pred):
    """
    Weighted Loss for Grid-Based Object Detection.
    y_true / y_pred shape: (batch, GRID_SIZE, GRID_SIZE, 5)
    Index 0: object confidence
    Index 1-2: dx, dy (cell offsets)
    Index 3-4: norm_w, norm_h (box dimensions)
    """
    obj_mask = y_true[..., 0:1]       # Shape: (batch, H, W, 1)
    noobj_mask = 1.0 - obj_mask

    # Confidence Loss (Binary Crossentropy)
    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    conf_loss = bce(y_true[..., 0:1], y_pred[..., 0:1])

    # Downweight empty background cells (0.5) vs object cells (5.0)
    weighted_conf_loss = (obj_mask * conf_loss * 5.0) + (noobj_mask * conf_loss * 0.5)

    # Box Coordinate Loss (Huber Loss, masked to positive object cells only)
    huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
    box_loss = huber(y_true[..., 1:5], y_pred[..., 1:5])
    masked_box_loss = box_loss * obj_mask

    total_conf = tf.reduce_mean(tf.reduce_sum(weighted_conf_loss, axis=[1, 2, 3]))
    total_box = tf.reduce_mean(tf.reduce_sum(masked_box_loss, axis=[1, 2, 3]))

    return total_conf + (total_box * 5.0)


def build_grid_model(img_size=512):
    """Builds ResNet50 with a custom 1x1 Conv Grid Head."""
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    # Output 5 channels (conf, dx, dy, w, h) via Sigmoid
    predictions = Conv2D(5, (1, 1), activation='sigmoid')(x)

    return Model(inputs=base_model.input, outputs=predictions)


def main():
    parser = argparse.ArgumentParser(description='Refactored Grid-Based ResNet Tree Detector')
    parser.add_argument('-d', '--datasets', nargs='+', required=True, help='Paths to dataset folders')
    parser.add_argument('-o', '--output', default='FineTunedResNetOD.keras', help='Output model path')
    parser.add_argument('-e', '--epochs', type=int, default=15, help='Training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size')
    args = parser.parse_args()

    IMG_SIZE = 512
    GRID_SIZE = 16

    matched_pairs = load_dataset_pairs(args.datasets)
    print(f"Found {len(matched_pairs)} matched image-annotation pairs.")

    if not matched_pairs:
        print("Error: No valid training pairs found.", file=sys.stderr)
        sys.exit(1)

    # Setup tf.data Pipeline
    output_signature = (
        tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(GRID_SIZE, GRID_SIZE, 5), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: make_dataset_generator(matched_pairs, IMG_SIZE, GRID_SIZE),
        output_signature=output_signature
    )

    dataset = dataset.shuffle(buffer_size=100).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # Build and Compile Model
    model = build_grid_model(img_size=IMG_SIZE)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=weighted_od_loss
    )

    model.summary()

    # Execute Training
    print("Starting Training...")
    model.fit(dataset, epochs=args.epochs)

    model.save(args.output)
    print(f"Model successfully saved to {args.output}")


if __name__ == '__main__':
    main()
