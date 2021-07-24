import torch

from model.CharacterClassification import CharacterClassification
from util.dataloader import dataloader
from util import helper

if torch.cuda.is_available():
    print("cuda enabled")
    DEVICE = torch.device('cuda')
else:
    print('cpu enabled')
    DEVICE = torch.devie('cpu')

print('Device :', DEVICE)
print("Torch version:{}".format(torch.__version__))
print("cuda version: {}".format(torch.version.cuda))
print("cudnn version:{}".format(torch.backends.cudnn.version()))

model = CharacterClassification().to(DEVICE)
print('==== model info ====')
print(model)
print('====================')

## no Train Model Save
model.eval()
trace_input = torch.rand(1, 784).to(DEVICE, dtype=torch.float32)
traced_script_module = torch.jit.trace(model, trace_input)
traced_script_module.save("C://Github//DeepLearningStudy//trained_model//NoTrainCharacterClassification.pt")
## no Train Model Save

loader_train = dataloader('C://Github//DeepLearningStudy//dataset//digits-train')

learning_rate = 0.001
batch_size = 10
sample_size = loader_train.sample_count()
total_batch = int(sample_size / batch_size)
target_accuracy = 0.95

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
cost = torch.nn.CrossEntropyLoss()


for epoch in range(200):
    average_cost = 0
    average_accuracy = 0
    for batch in range(total_batch):
        inputs, labels = loader_train.load([784], [10], 1.0, batch_size, False)
        if inputs is None or labels is None:
            break

        batches = (inputs, labels)
        current_cost, current_accuracy = helper.train(model, optimizer, cost, batches, DEVICE)
        average_cost += (current_cost / total_batch)
        average_accuracy += (current_accuracy / total_batch)

    loader_train.clear()

    print('epoch :', epoch, ', cost:', average_cost, ", accuracy:", average_accuracy)
    if average_accuracy > target_accuracy:
        break

model.eval()
trace_input = torch.rand(1, 784).to(DEVICE, dtype=torch.float32)
traced_script_module = torch.jit.trace(model, trace_input)
traced_script_module.save("C://Github//DeepLearningStudy//trained_model//CharacterClassification.pt")
