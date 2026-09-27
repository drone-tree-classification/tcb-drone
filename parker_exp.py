import cv2

from ultralytics import YOLO

# 1. Load a pre-trained YOLOv8 model (trained on the 80-class COCO dataset)

# It will automatically download the file 'yolov8n.pt' on your first run

model = YOLO('yolov8n.pt')

# 2. Open your video file or webcam stream

video_path = '/home/ryan/code/tcb-drone/SampleFlyover.MP4'

cap = cv2.VideoCapture(video_path)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:

        break  # End of video

    # 3. Run inference on the current frame

    # persist=True tracks objects smoothly across frames

    results = model(frame, stream=True)

    for result in results:

        # Get coordinates, class IDs, and confidence scores

        boxes = result.boxes.xyxy.cpu().numpy()     # Box coordinates [xmin, ymin, xmax, ymax]

        classes = result.boxes.cls.cpu().numpy()   # Class index integers

        scores = result.boxes.conf.cpu().numpy()   # Confidence levels (0.0 - 1.0)

        names = result.names                       # Dictionary mapping ID to string label

        # 4. Loop through detected objects and draw them

        for box, cls_idx, score in zip(boxes, classes, scores):

            if score > 0.4:  # Only draw boxes if confidence is greater than 40%

                xmin, ymin, xmax, ymax = map(int, box)

                label = f"{names[int(cls_idx)]}: {score:.2f}"

                # Draw bounding box rectangle

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Draw label background and text string

                cv2.putText(frame, label, (xmin, ymin - 10),

                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 5. Display the processed frame

    cv2.imshow('Pre-trained Object Detection', frame)

    # Break out of loop if 'q' key is pressed

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

cap.release()

cv2.destroyAllWindows()
