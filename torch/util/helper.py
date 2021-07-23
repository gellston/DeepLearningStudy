import torch



def train(model, optimizer, cost_function, batches, device):
    input = torch.from_numpy(batches[0]).to(device, dtype=torch.float32)
    label = torch.from_numpy(batches[1]).to(device, dtype=torch.long)

    # Cost
    model.train()
    optimizer.zero_grad()  ## gradient clear
    output = model(input)  ## Model output
    cost = cost_function(output, label)  ## loss calculation
    cost.backward()  ## loss gradient calculation
    cost = cost.item()
    optimizer.step()  ## update weight

    # Accuracy
    model.eval()
    output = model(input)
    prediction = output.max(1, keepdim=True)[1]
    accuracy = prediction.eq(label.view_as(prediction)).sum()
    accuracy = accuracy.item() / len(batches[1])


    return (cost, accuracy)


