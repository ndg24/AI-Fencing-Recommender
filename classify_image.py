"""Classify an image against ImageNet classes using InceptionV3.

Modernized replacement for the original script, which downloaded a frozen
TF1 graph and relied on tf.app.flags/tf.gfile APIs that no longer exist in
current TensorFlow. This uses tf.keras.applications instead, which fetches
pretrained InceptionV3 weights automatically.
"""

import argparse

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import (
    InceptionV3,
    decode_predictions,
    preprocess_input,
)
from tensorflow.keras.preprocessing import image as keras_image

_feature_extractor = None


def build_feature_extractor():
    """InceptionV3 with the classification head removed, pooled to a 2048-d vector.

    This is the modern equivalent of the old frozen graph's 'pool_3:0' tensor
    that preloaded_inception.py used to read features from.
    """
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    return _feature_extractor


def extract_features_from_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (299, 299))
    frame_array = np.expand_dims(frame_resized.astype("float32"), axis=0)
    frame_array = preprocess_input(frame_array)
    features = build_feature_extractor().predict(frame_array, verbose=0)
    return features[0]


def run_inference_on_image(image_path, num_top_predictions=5):
    model = InceptionV3(weights="imagenet")

    img = keras_image.load_img(image_path, target_size=(299, 299))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)
    decoded = decode_predictions(predictions, top=num_top_predictions)[0]

    for _, human_string, score in decoded:
        print("%s (score = %.5f)" % (human_string, score))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_file", required=True, help="Absolute path to image file."
    )
    parser.add_argument(
        "--num_top_predictions",
        type=int,
        default=5,
        help="Display this many predictions.",
    )
    args = parser.parse_args()

    run_inference_on_image(args.image_file, args.num_top_predictions)


if __name__ == "__main__":
    main()
