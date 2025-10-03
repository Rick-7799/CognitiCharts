import tensorflow as tf
def make_image_model(img_size=224, num_classes=3):
    I=tf.keras.Input(shape=(img_size,img_size,3))
    x=tf.keras.layers.Conv2D(32,3,activation='relu',padding='same')(I); x=tf.keras.layers.MaxPool2D()(x)
    x=tf.keras.layers.Conv2D(64,3,activation='relu',padding='same')(x); x=tf.keras.layers.MaxPool2D()(x)
    x=tf.keras.layers.Conv2D(128,3,activation='relu',padding='same')(x)
    x=tf.keras.layers.GlobalAveragePooling2D()(x); x=tf.keras.layers.Dropout(0.3)(x)
    x=tf.keras.layers.Dense(128,activation='relu')(x); x=tf.keras.layers.Dropout(0.15)(x)
    O=tf.keras.layers.Dense(num_classes,activation='softmax')(x)
    m=tf.keras.Model(I,O); m.compile(optimizer=tf.keras.optimizers.Adam(5e-4),loss='sparse_categorical_crossentropy',metrics=['accuracy']); return m
