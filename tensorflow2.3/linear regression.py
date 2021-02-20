import numpy as np
from model.SimpleDense import SimpleDense
from tensorflow.keras.models import model_from_json

x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).reshape(10, 1)
y_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).reshape(10, 1)

model = SimpleDense(learning_rate=0.001)

for step in range(500):
  print("============================================")
  print("current step = ", step)
  loss_score = model.train_one_batch(x_data, y_data)
  output = model.predict(np.array([1]))
  print('result = ')
  print(output)
  print("============================================")
  print('loss score = ', loss_score)

#model.get_model().save("C:\\Github\\DeepLearningStudy\\trained_model\\")

# Save the weights
model.get_model().save_weights("C:\\Github\\DeepLearningStudy\\trained_model\\SimpleDense.h5")

# Save the model architecture as json
with open("C:\\Github\\DeepLearningStudy\\trained_model\\SimpleDense.json", "w") as fp:
  fp.write(model.get_model().to_json(indent="\t"))



# Reconfigure model in json file
with open("C:\\Github\\DeepLearningStudy\\trained_model\\SimpleDense.json", "r") as fp:
  load_test_model = model_from_json(fp.read())

# Load weights to new model
load_test_model.load_weights("C:\\Github\\DeepLearningStudy\\trained_model\\SimpleDense.h5")

print('script end')