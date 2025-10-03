import tensorflow as tf
def make_tf_model(seq_len, num_features, num_classes=3):
    inputs = tf.keras.Input(shape=(seq_len, num_features))
    x = tf.keras.layers.Conv1D(48, 5, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(96, 5, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(96, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
