import numpy as np
import tensorflow as tf
from model.SelfieSegmentation import SelfieSegmentation
from tensorflow.keras.models import model_from_json
from util.segmentation_dataloader import segmentation_dataloader


loader_train = segmentation_dataloader('C://Github//DeepLearningStudy//dataset//portrait_segmentation_input256x256//',
                                       'C://Github//DeepLearningStudy//dataset//portrait_segmentation_label256x256//')


learning_rate = 0.003
batch_size = 10
sample_size = loader_train.size()
total_batch = int(sample_size / batch_size)
target_accuracy = 0.97

model = SelfieSegmentation(learning_rate=learning_rate)
tf.keras.utils.plot_model(model.get_model(), to_file='C:\\Github\\DeepLearningStudy\\trained_model\\SelfieSegmentation.png', show_shapes=True, show_layer_names=True)

for epoch in range(200):
    average_cost = 0
    average_accuracy = 0
    for batch in range(total_batch):
        inputs_train, outputs_train = loader_train.load([256, 256, 3], [256, 256, 1], 1, 255, batch_size)
        if inputs_train is None or outputs_train is None:
            break

        cost = model.train_one_batch(inputs_train, outputs_train)
        accuracy = model.accracy_on_batch(inputs_train, outputs_train)
        average_cost += (cost / total_batch)
        average_accuracy += (accuracy / total_batch)

    loader_train.clear()

    print('current epoch = ', epoch)
    print('average cost = ', average_cost)
    print('average accuracy = ', average_accuracy)

    if average_accuracy > target_accuracy:
        break;


tf.saved_model.save(model.get_model(), "C:\\Github\\DeepLearningStudy\\trained_model\\SelfieSegmentation")

# Save the weights
model.get_model().save_weights("C:\\Github\\DeepLearningStudy\\trained_model\\SelfieSegmentation.h5")

# Save the model architecture as json
with open("C:\\Github\\DeepLearningStudy\\trained_model\\SelfieSegmentation.json", "w") as fp:
  fp.write(model.get_model().to_json(indent="\t"))


