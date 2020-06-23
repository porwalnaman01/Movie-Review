import tensorflow_datasets as tfds
import tensorflow as tf
import os
import tensorflow_hub as hub


def predictValue(array):
    results = []
    for elem in array:
        if elem > 0.5:
            results.append('positive')
        else:
            results.append('negative')
    return results


train_data, val_data, test_data = tfds.load(name='imdb_reviews',
                                            split=['train[:60%]', 'train[60%:]', 'test'],
                                            as_supervised=True)

train_example_batch, train_label_batch = next(iter(train_data.batch(10)))
# print(train_example_batch)

pretrained_model = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(pretrained_model, input_shape=[],
                           dtype=tf.string, trainable=True)

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


checkpoint_path = r"C:\Users\HP\PycharmProjects\machine learning\training_1\cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights so that if computer crashes we have already saved them
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
# Train the model with the new callback
model.fit(train_data.shuffle(10000).batch(512),
          epochs=20,
          validation_data=val_data.batch(512),
          verbose=1,
          callbacks=[cp_callback])

prediction = model.predict(['any movie review'])
print(predictValue([prediction]))
