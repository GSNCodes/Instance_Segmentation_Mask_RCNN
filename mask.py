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


class MyConfig(CocoConfig):
    NAME = "my_coco_inference"
    # Set batch size to 1 since we'll be running inference on one image at a time.
    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def prepare_mrcnn_model(model_path, model_name, class_names, my_config):
    classes = open(class_names).read().strip().split("\n")
    print("No. of classes", len(classes))

    hsv = [(i / len(classes), 1, 1.0) for i in range(len(classes))]
    COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.seed(42)
    random.shuffle(COLORS)

    model = modellib.MaskRCNN(mode="inference", model_dir=model_path, config=my_config)
    model.load_weights(model_name, by_name=True)

    return COLORS, model, classes


def perform_inference_image(image_path, model, colors, classes, draw_bbox, mrcnn_visualize, instance_segmentation,
                            save_enable):
    test_image = cv2.imread(image_path)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    output = custom_visualize(test_image, model, colors, classes, draw_bbox, mrcnn_visualize, instance_segmentation)
    if not mrcnn_visualize:
        if save_enable:
            cv2.imwrite("result.png", output)
        cv2.imshow("Output", output)
        cv2.waitKey()
        cv2.destroyAllWindows()


def custom_visualize(test_image, model, colors, classes, draw_bbox, mrcnn_visualize, instance_segmentation):
    detections = model.detect([test_image], verbose=1)[0]

    if mrcnn_visualize:
        matplotlib.use('TkAgg')
        visualize.display_instances(test_image, detections['rois'], detections['masks'], detections['class_ids'],
                                    classes,
                                    detections['scores'])

        return

    if instance_segmentation:
        hsv = [(i / len(detections['rois']), 1, 1.0) for i in range(len(detections['rois']))]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(42)
        random.shuffle(colors)

    for i in range(0, detections["rois"].shape[0]):
        classID = detections["class_ids"][i]

        mask = detections["masks"][:, :, i]
        if instance_segmentation:
            color = colors[i][::-1]
        else:
            color = colors[classID][::-1]

        # To visualize the pixel-wise mask of the object
        test_image = visualize.apply_mask(test_image, mask, color, alpha=0.5)

    test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

    if draw_bbox == 'True':
        for i in range(0, len(detections["scores"])):
            (startY, startX, endY, endX) = detections["rois"][i]

            classID = detections["class_ids"][i]
            label = classes[classID]
            score = detections["scores"][i]

            if instance_segmentation:
                color = [int(c) for c in np.array(colors[i]) * 255]

            else:
                color = [int(c) for c in np.array(colors[classID]) * 255]

            cv2.rectangle(test_image, (startX, startY), (endX, endY), color, 2)
            text = "{}: {:.2f}".format(label, score)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(test_image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return test_image


def perform_inference_video(use_camera, video_path, model, colors, classes, draw_bbox, mrcnn_visualize,
                            instance_segmentation, save_enable):
    if use_camera:
        video = cv2.VideoCapture(0)
        time.sleep(2.0)
    else:
        video = cv2.VideoCapture(video_path)

    video_flag = True
    while True:
        ret, frame = video.read()

        if save_enable and video_flag:
            out = cv2.VideoWriter("Result_Video.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 8,
                                  (frame.shape[1], frame.shape[0]))
            video_flag = False

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        output = custom_visualize(frame, model, colors, classes, draw_bbox, mrcnn_visualize, instance_segmentation)

        cv2.imshow("Output", output)

        if save_enable:
            out.write(output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    video.release()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Path to the test images', default=None)
    parser.add_argument('--model_path', help='Path to the model directory', default='models/')
    parser.add_argument('--model_name', help='Name of the model file', default='models/mask_rcnn_coco.h5')
    parser.add_argument('--class_names', help='Path to the class labels', default='coco_classes.txt')
    parser.add_argument('--mrcnn_visualize', help='Use the built-in visualize method', type=bool, default=False)
    parser.add_argument('--instance_segmentation', help='To toggle between semantic and instance segmentation',
                        type=bool, default=True)
    parser.add_argument('--draw_bbox', help='Draw the bounding box with class labels', type=bool, default=True)
    parser.add_argument('--camera', help='Perform live detection', type=bool, default=False)
    parser.add_argument('--video', help='Path to video file', default=None)
    parser.add_argument('--save_enable', help='Enable to save processed image or video', type=bool, default=False)
    args = vars(parser.parse_args())

    if args['image']:
        my_config = MyConfig()
        my_config.display()
        colors, model, classes = prepare_mrcnn_model(args['model_path'], args['model_name'], args['class_names'],
                                                     my_config)
        perform_inference_image(args['image'], model, colors, classes, args['draw_bbox'], args['mrcnn_visualize'],
                                args['instance_segmentation'], args['save_enable'])

    if args['camera'] or args['video']:
        use_camera = args['camera']
        video_path = args['video']

        my_config = MyConfig()
        my_config.display()
        colors, model, classes = prepare_mrcnn_model(args['model_path'], args['model_name'], args['class_names'],
                                                     my_config)
        perform_inference_video(use_camera, video_path, model, colors, classes, args['draw_bbox'],
                                args['mrcnn_visualize'],
                                args['instance_segmentation'], args['save_enable'])
