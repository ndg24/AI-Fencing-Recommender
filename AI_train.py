import os

import hickle as hkl
import numpy as np
import tensorflow as tf
from tensorflow import keras

num_layers = 4
n_hidden = 64
drop_out_rate = 0.2
batch_size = 4
epochs = 3
learning_rate = 1e-4
n_classes = 3


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def build_model(sequence_length, n_input):
    inputs = keras.Input(shape=(sequence_length, n_input))
    x = inputs
    for layer_index in range(num_layers):
        return_sequences = layer_index < num_layers - 1
        x = keras.layers.LSTM(n_hidden, return_sequences=return_sequences, dropout=drop_out_rate)(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    loaded = None
    videos_loaded = 0
    for i in os.listdir(os.getcwd()):
        if i.endswith(".hkl") and "features" in i:
            print(i)
            data = hkl.load(i)
            loaded = data if videos_loaded == 0 else np.concatenate((loaded, data), axis=0)
            videos_loaded += 1
            print(loaded.shape)

    labels = None
    videos_loaded = 0
    for i in os.listdir(os.getcwd()):
        if i.endswith(".hkl") and "labels" in i:
            print(i)
            data = hkl.load(i)
            labels = data if videos_loaded == 0 else np.concatenate((labels, data), axis=0)
            videos_loaded += 1
            print(labels.shape)

    if loaded is None or labels is None:
        print("No *features*.hkl / *labels*.hkl files found in the current directory.")
        return

    loaded, labels = unison_shuffled_copies(loaded, labels)
    print(loaded.shape, labels.shape)

    n_examples = len(loaded)
    test_size = max(1, int(n_examples * 0.15))
    validation_size = max(1, int(n_examples * 0.15))

    test_set = loaded[:test_size]
    test_labels = labels[:test_size]
    validation_set = loaded[test_size : test_size + validation_size]
    validation_labels = labels[test_size : test_size + validation_size]
    train_set = loaded[test_size + validation_size :]
    train_labels = labels[test_size + validation_size :]

    print("Test Set Shape:", test_set.shape)
    print("Validation Set Shape:", validation_set.shape)
    print("Training Set Shape:", train_set.shape)

    hkl.dump(test_set, "test_data.hkl", mode="w", compression="gzip")
    hkl.dump(test_labels, "test_lbls.hkl", mode="w", compression="gzip")

    n_input = train_set.shape[-1]
    model = build_model(train_set.shape[1], n_input)
    model.summary()

    validation_data = (validation_set, validation_labels) if len(validation_set) > 0 else None
    model.fit(
        train_set,
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
    )

    model.save("fencing_ai_model.keras")

    if len(test_set) > 0:
        test_loss, test_acc = model.evaluate(test_set, test_labels)
        print("Test Accuracy:", test_acc)

    print("Learning finished!")


if __name__ == "__main__":
    main()
