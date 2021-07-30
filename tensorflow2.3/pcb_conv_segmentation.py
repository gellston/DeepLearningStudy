import numpy as np
import tensorflow as tf
import pydotplus
import cv2


from model.PCBDefectSegmentation import PCBDefectSegmentation
from model.PCBDefectSegmentationV7 import PCBDefectSegmentationV7
from tensorflow.keras.models import model_from_json
from util.opendl_segmentation_dataloader import opendl_segmentation_dataloader


print(tf.__version__)


loader_train = opendl_segmentation_dataloader('C://Users//gellston//Desktop//PCB_Augmentation_Final_512_Rotation//')
loader_test = opendl_segmentation_dataloader('C://Users//gellston//Desktop//PCB_Augmentation_Final_512_Rotation//')

learning_rate = 0.1
batch_size = 4
sample_size = loader_train.size()
total_batch = int(sample_size / batch_size)
target_accuracy = 0.94

model = PCBDefectSegmentationV7(learning_rate=learning_rate)
tf.keras.utils.plot_model(model.get_model(), to_file='C:\\Github\\DeepLearningStudy\\trained_model\\PCBDefectSegmentation.png', show_shapes=True, show_layer_names=True)
for epoch in range(1000):
    average_cost = 0
    average_accuracy = 0
    for batch in range(total_batch):
        inputs_train, outputs_train = loader_train.load([512, 512, 3], [512, 512, 1], 1, 255, batch_size)
        if inputs_train is None or outputs_train is None:
            break

        cost = model.train_one_batch(inputs_train, outputs_train)
        accuracy = model.accracy_on_batch(inputs_train, outputs_train)
        average_cost += (cost / total_batch)
        average_accuracy += (accuracy / total_batch)


    loader_test.clear()
    loader_test.shuffle()
    input_test, output_test = loader_test.load([512, 512, 3], [512, 512, 1], 1, 255, 1)
    output_test = model.predict(input_test)
    output_test = output_test[0] * 255;
    output_test = output_test.astype(np.uint8)
    _, threshold = cv2.threshold(output_test, 204, 255, cv2.THRESH_BINARY)
    ##visual_output = (np.array(threshold) * 255).astype('uint8') ;

    cv2.imshow('output', threshold)
    cv2.waitKey(33)

    loader_train.clear()
    loader_train.shuffle()

    print('current epoch = ', epoch)
    print('average cost = ', average_cost)
    print('average accuracy = ', average_accuracy)

    if average_accuracy > target_accuracy:
        break;


tf.saved_model.save(model.get_model(), "C:\\Github\\DeepLearningStudy\\trained_model\\PCBDefectSegmentation")

# Save the weights
model.get_model().save_weights("C:\\Github\\DeepLearningStudy\\trained_model\\PCBDefectSegmentation.h5")

# Save the model architecture as json
with open("C:\\Github\\DeepLearningStudy\\trained_model\\PCBDefectSegmentation.json", "w") as fp:
  fp.write(model.get_model().to_json(indent="\t"))


