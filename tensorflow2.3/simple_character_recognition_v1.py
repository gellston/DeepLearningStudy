import numpy as np
import tensorflow as tf
from model.SimpleDenseRecogCharacter import SimpleDenseRecogCharacter
from tensorflow.keras.models import model_from_json
from util.dataloader import dataloader


loader_train = dataloader('C://Github//DeepLearningStudy//dataset//digits-train')


learning_rate = 0.001
batch_size = 1000
sample_size = loader_train.sample_count()
total_batch = int(sample_size / batch_size)
target_accuracy = 0.90

model = SimpleDenseRecogCharacter(learning_rate=learning_rate)


for epoch in range(1000):
    average_cost = 0
    average_accuracy = 0
    for batch in range(total_batch):
        inputs_train, outputs_train = loader_train.load([28, 28, 1], 1.0, batch_size, False)
        if inputs_train is None or outputs_train is None:
            loader_train.clear()
            break

        cost = model.train_one_batch(inputs_train, outputs_train)
        accuracy = model.accracy_on_batch(inputs_train, outputs_train)
        average_cost += cost / total_batch
        average_accuracy += accuracy / total_batch


    loader_train.clear()
    print('current epoch = ', epoch)
    print('average cost = ', average_cost)
    print('average accuracy = ', average_accuracy)





