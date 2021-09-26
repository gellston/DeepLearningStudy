import numpy as np
import tensorflow as tf
from model.LeafletLineClassification import LeafletLineClassification
from tensorflow.keras.models import model_from_json
from util.dataloader import dataloader


loader_train = dataloader('C://Github//LeafletImageCropTool//Data//ClassificationExperiment')
loader_train.shuffle()

learning_rate = 0.005
batch_size = 40
sample_size = loader_train.sample_count()
total_batch = int(sample_size / batch_size)
target_accuracy = 0.9997

model = LeafletLineClassification(learning_rate=learning_rate)
tf.keras.utils.plot_model(model.get_model(), to_file='C:\\Github\\DeepLearningStudy\\trained_model\\LeafletLineClassification.png', show_shapes=True, show_layer_names=True)

for epoch in range(200):
    average_cost = 0
    average_accuracy = 0
    for batch in range(total_batch):
        inputs_train, outputs_train = loader_train.load([100, 512, 3], 1, batch_size, True)
        if inputs_train is None or outputs_train is None:
            break

        cost = model.train_one_batch(inputs_train, outputs_train)
        accuracy = model.accracy_on_batch(inputs_train, outputs_train)
        average_cost += (cost / total_batch)
        average_accuracy += (accuracy / total_batch)



    loader_train.clear()
    #loader_train.shuffle()

    print('current epoch = ', epoch)
    print('average cost = ', average_cost)
    print('average accuracy = ', average_accuracy)

    if average_accuracy >= target_accuracy:
        break;


# Save the weights
model.get_model().save_weights("C:\\Github\\DeepLearningStudy\\trained_model\\LeafletLineClassification.h5")

# Save the model architecture as json
with open("C:\\Github\\DeepLearningStudy\\trained_model\\LeafletLineClassification.json", "w") as fp:
  fp.write(model.get_model().to_json(indent="\t"))



