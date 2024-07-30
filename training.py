from utils.config import *
from utils.load_data import *
from utils.show_image import *

import tensorflow as tf
import keras
from keras_cv import bounding_box
from keras_cv import visualization
from keras_cv.callbacks import PyCOCOCallback

import numpy as np
import cv2
import matplotlib.pyplot as plt



# Load data
# ("image": width*height,
#  "bounding_boxes":{
#               'boxes': Tensor(shape=[batch, num_boxes, 4]),
#               'classes': Tensor(shape=[batch, num_boxes])
#                    })
train_ds, val_ds = load_data()


# conver data to tensor not tuple
train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


# we have chose backbone in: https://keras.io/api/keras_cv/models/
#   yolo_v8_xs_backbone_coco: 1.28M, 
#   yolo_v8_s_backbone_coco: 5.09M, 
#   yolo_v8_m_backbone_coco: 11.87M
#   yolo_v8_l_backbone_coco: 19.83M
#   yolo_v8_xl_backbone_coco: 30.97M

backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_xl_backbone_coco"  # We will use yolov8 small backbone with coco weights
)


yolo = keras_cv.models.YOLOV8Detector(
    num_classes=len(class_mapping),
    bounding_box_format="xyxy",
    backbone=backbone,
    fpn_depth=1,
)



optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    global_clipnorm=GLOBAL_CLIPNORM,
)

yolo.compile(
    optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou"
)

callback = PyCOCOCallback(
    validation_data=val_ds,
    bounding_box_format="xyxy",
    )


yolo.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1000,
    callbacks=callback,
)


yolo.save("model/model.keras")