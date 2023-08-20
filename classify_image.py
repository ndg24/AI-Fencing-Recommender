"""Classify an image against ImageNet classes using InceptionV3.

Modernized replacement for the original script, which downloaded a frozen
TF1 graph and relied on tf.app.flags/tf.gfile APIs that no longer exist in
current TensorFlow. This uses tf.keras.applications instead, which fetches
pretrained InceptionV3 weights automatically.
"""

import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import (
    InceptionV3,
    decode_predictions,
    preprocess_input,
)
from tensorflow.keras.preprocessing import image as keras_image


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
