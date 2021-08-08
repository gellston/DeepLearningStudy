import torch

from torchsummary import summary

from model.AnimalClassificationV1 import AnimalClassificationV1
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

model = AnimalClassificationV1().to(DEVICE)
model = model.to(memory_format=torch.channels_last)

##print('check channel order = ', model.is_contiguous(memory_format=torch.channels_last))
print('==== model info ====')
summary(model, (100, 100, 3))
print('====================')

## no Train Model Save
model.eval()
trace_input = torch.rand(1, 100, 100, 3).to(DEVICE, dtype=torch.float32)
traced_script_module = torch.jit.trace(model, trace_input)
traced_script_module.save("C://Github//DeepLearningStudy//trained_model//NoTrainAnimalClassificationV1.pt")
## no Train Model Save

loader_train = dataloader('C://Github//DeepLearningStudy//dataset//animal-train-v1')
loader_valid = dataloader('C://Github//DeepLearningStudy//dataset//animal-validation-v1')

learning_rate = 0.001
batch_size = 10
sample_size = loader_train.sample_count()
total_batch = int(sample_size / batch_size)
target_accuracy = 0.86

valid_sample_size = loader_valid.sample_count()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
cost = torch.nn.CrossEntropyLoss()


for epoch in range(200):
    average_cost = 0
    train_average_accuracy = 0
    valid_average_accuracy = 0
    for batch in range(total_batch):
        ## Loader에 배치 사이즈를 넣지 않아도 되도록 수정 필요
        inputs, labels = loader_train.load([100, 100, 3], [batch_size, 1, 1], 1.0, batch_size, True)
        if inputs is None or labels is None:
            break

        batches = (inputs, labels)
        current_cost, current_accuracy = helper.train(model, optimizer, cost, batches, DEVICE)
        average_cost += (current_cost / total_batch)
        train_average_accuracy += (current_accuracy / total_batch)

    loader_train.clear()

    for batch in range(valid_sample_size):
        ## Loader에 배치 사이즈를 넣지 않아도 되도록 수정 필요
        inputs, labels = loader_valid.load([100, 100, 3], [1, 1, 1], 1.0, 1, True)
        if inputs is None or labels is None:
            break
        batches = (inputs, labels)
        _, current_accuracy = helper.train(model, optimizer, cost, batches, DEVICE)
        valid_average_accuracy += current_accuracy

    ## Valid accuracy가 나오지 않음. 확인 필요
    valid_average_accuracy = valid_average_accuracy / valid_sample_size

    print('epoch :', epoch, ', cost:', average_cost, ", train accuracy:", train_average_accuracy)
    print('== valid accuracy = ', valid_average_accuracy)

    if train_average_accuracy > target_accuracy:
        break

model.eval()
trace_input = torch.rand(1, 100, 100, 3).to(DEVICE, dtype=torch.float32)
traced_script_module = torch.jit.trace(model, trace_input)
traced_script_module.save("C://Github//DeepLearningStudy//trained_model//AnimalClassificationV1.pt")
