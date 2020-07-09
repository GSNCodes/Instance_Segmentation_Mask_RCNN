import cv2
import os
import numpy as np
import random
import colorsys
import argparse
import time
from mrcnn import model as modellib
from mrcnn import visualize
from samples.coco.coco import CocoConfig
import matplotlib

ap = argparse.ArgumentParser()
ap.add_argument('--image', help='Path to the test images', default=None)
ap.add_argument('--model_path', help='Path to the model directory', default='models/')
ap.add_argument('--model_name', help='Name of the model file', default='models/mask_rcnn_coco.h5')
ap.add_argument('--class_names', help='Path to the class labels', default='coco_classes.txt')
ap.add_argument('--mrcnn_visualize', help='Use the built-in visualize method', type=bool, default=False)
ap.add_argument('--draw_bbox', help='Draw the bounding box with class labels', type=bool, default=True)
ap.add_argument('--camera', help='Perform live detection', type=bool, default=False)
args = vars(ap.parse_args())

# def main():
    # mrcnn_visualize = False
    # show_bbox = True


class MyConfig(CocoConfig):
    NAME = "my_coco_inference"
    # Set batch size to 1 since we'll be running inference on one image at a time.
    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


my_config = MyConfig()
my_config.display()



classes = open(args['class_names']).read().strip().split("\n")
print("No. of classes", len(classes))

hsv = [(i / len(classes), 1, 1.0) for i in range(len(classes))]
COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
random.seed(42)
random.shuffle(COLORS)


model = modellib.MaskRCNN(mode="inference", model_dir=args['model_path'], config=my_config)
model.load_weights(args['model_name'], by_name=True)

test_image = cv2.imread(args['image'])
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

result = model.detect([test_image], verbose=1)[0]

if args['mrcnn_visualize']:
    matplotlib.use('TkAgg')
    visualize.display_instances(test_image, result['rois'], result['masks'], result['class_ids'], classes,
                                result['scores'])



for i in range(0, result["rois"].shape[0]):
    classID = result["class_ids"][i]

    mask = result["masks"][:, :, i]
    color = COLORS[classID][::-1]

    # To visualize the pixel-wise mask of the object
    test_image = visualize.apply_mask(test_image, mask, color, alpha=0.5)


test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)


if args['draw_bbox']:
    for i in range(0, len(result["scores"])):
        (startY, startX, endY, endX) = result["rois"][i]

        classID = result["class_ids"][i]
        label = classes[classID]
        score = result["scores"][i]
        color = [int(c) for c in np.array(COLORS[classID]) * 255]

        cv2.rectangle(test_image, (startX, startY), (endX, endY), color, 2)
        text = "{}: {:.3f}".format(label, score)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(test_image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Display the output image
cv2.imshow("Output", test_image)
cv2.waitKey()

# if __name__ == '__main__':
    # ap = argparse.ArgumentParser()
    # ap.add_argument('--image', help='Path to the test images', default=None)
    # ap.add_argument('--model_path', help='Path to the model directory', default='models/')
    # ap.add_argument('--model_name', help='Name of the model file', default='mask_rcnn_coco.h5')
    # ap.add_argument('--class_names', help='Path to the class labels', default='coco_classes.txt')
    # ap.add_argument('--mrcnn_visualize', help='Use the built-in visualize method', type=bool, default=False)
    # ap.add_argument('--draw_bbox', help='Draw the bounding box with class labels', type=bool, default=True)

    # main()


if args['camera']:
    video = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        ret, image = video.read()
        result = model.detect([image], verbose=1)[0]

        for i in range(0, result["rois"].shape[0]):
            classID = result["class_ids"][i]

            mask = result["masks"][:, :, i]
            color = COLORS[classID][::-1]

            # To visualize the pixel-wise mask of the object
            test_image = visualize.apply_mask(test_image, mask, color, alpha=0.5)


        test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)


        if args['draw_bbox']:
            for i in range(0, len(result["scores"])):
                (startY, startX, endY, endX) = result["rois"][i]

                classID = result["class_ids"][i]
                label = classes[classID]
                score = result["scores"][i]
                color = [int(c) for c in np.array(COLORS[classID]) * 255]

                cv2.rectangle(test_image, (startX, startY), (endX, endY), color, 2)
                text = "{}: {:.3f}".format(label, score)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(test_image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)