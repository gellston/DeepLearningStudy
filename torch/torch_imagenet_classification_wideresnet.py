import torch
import torchvision
import torchvision.transforms as transforms

from model.WideResNet import WideResNet

# Set device
USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

## Hyper parameter
training_epochs = 10
batch_size = 30
target_accuracy = 0.70
learning_rate = 0.003
num_class = 1000
save_step_batch_size = 1000
skip_batch_count = 0
pretrained = False
accuracy_threshold = 0.65
## Hyper parameter

# Initialize transformations for data augmentation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Load the ImageNet Object Localization Challenge dataset
train_dataset = torchvision.datasets.ImageFolder(root='D://training_image//imagenet//train//',
                                                 transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load the ResNet50 model
model = WideResNet(class_num=num_class)

# Parallelize training across multiple GPUs

# Set the model to run on the device
model = model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)


top_accuracy = 0

# Train the model...
for epoch in range(training_epochs):
    avg_cost = 0
    avg_acc = 0
    current_batch = 0
    for inputs, labels in train_loader:
        # Move input and label tensors to the device
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = torch.nn.functional.one_hot(labels, num_classes=num_class).float()

        # Zero out the optimizer
        optimizer.zero_grad()

        # Forward pass
        model.train()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        avg_cost += (loss.item() / batch_size)

        # Backward pass
        loss.backward()
        optimizer.step()

        model.eval()
        hypothesis = model(inputs)
        correct_prediction = torch.argmax(hypothesis, 1) == torch.argmax(labels, 1)
        accuracy = correct_prediction.float().mean().item()
        avg_acc += (accuracy / batch_size)

        current_batch += 1
        if current_batch % save_step_batch_size == 0:
            ## no Train Model Save
            model.eval()
            compiled_model = torch.jit.script(model)
            torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//ImageNet(WideResnet_Batch_Step).pt")
            gc.collect()
            ## no Train Model Save
            print('current batch=', current_batch, 'current accuracy=', avg_acc, 'current cost=', avg_cost)


    # Print the loss for every epoch
    print(f'Epoch {epoch+1}/{num_class}, Loss: {avg_cost:.4f}, Accuracy: {avg_acc:.4f}')
    if top_accuracy < avg_acc:
        top_accuracy = avg_acc
        ## no Train Model Save
        model.eval()
        compiled_model = torch.jit.script(model)
        torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//ImageNet(WideResnet_TOP).pt")
        gc.collect()
        ## no Train Model Save

    if avg_acc >= target_accuracy:
        break


## no Train Model Save
model.eval()
compiled_model = torch.jit.script(model)
torch.jit.save(compiled_model, "C://Github//DeepLearningStudy//trained_model//ImageNet(WideResnet_Over).pt")
gc.collect()
## no Train Model Save
print('current batch=', current_batch, 'current accuracy=', avg_acc, 'current cost=', avg_cost)

print(f'Finished Training, Loss: {loss.item():.4f}')