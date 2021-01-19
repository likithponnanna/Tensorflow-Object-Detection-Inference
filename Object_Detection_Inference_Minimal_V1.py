#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import pathlib
tf.get_logger().setLevel('ERROR')
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from obj_det.utils import ops as utils_ops
from obj_det.utils import label_map_util

# Helper function to extract id to value map from labelmap.
def build_id_class_map(d):
    for i in d:
        tempKey = None
        count = 0
        for j, k in d[i].items():
            if count == 0:
                tempKey = k
                count+= 1
            else:
                id_to_value_map[tempKey] = k
                count = 0
                
def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
    # Handle models with masks:
    if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    return output_dict

def show_inference(model, image_path):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.

    per_image_detection = dict()
    for i, detected_class_id in np.ndenumerate(output_dict['detection_classes']):
        if id_to_value_map[detected_class_id] in per_image_detection and  output_dict['detection_scores'][i] > 0.5:
            per_image_detection[id_to_value_map[detected_class_id]].append('{:.2f}%'.format(output_dict['detection_scores'][i] * 100))
        elif output_dict['detection_scores'][i] > 0.5:
            per_image_detection[id_to_value_map[detected_class_id]] = ['{:.2f}%'.format(output_dict['detection_scores'][i] * 100)]

    return per_image_detection
                
                
#Load the saved model.
print("Model being Loaded!! Please wait it will take time for the model to load.")
detection_model = tf.saved_model.load('obj_det/Inference_Graph/saved_model')
print("Model load completed!!")


# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'obj_det/Label_Map/labelmap.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('obj_det/Test_Images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*")))
id_to_value_map = dict()
          
build_id_class_map(category_index)
final_dict = dict()

# Runing inference image by by image.
for image_path in TEST_IMAGE_PATHS:
    final_dict[str(image_path)] = show_inference(detection_model, image_path)


# Json Print and dumping the json output file from the output dictionary.
final_json  = json.dumps(final_dict, indent=4, sort_keys=True)
print("Final Result: ", final_json)
with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(final_dict, f, ensure_ascii=False, indent=4)








