import torch


if torch.cuda.is_available():
    print("cuda enabled")
    DEVICE = torch.device('cuda')
else:
    print('cpu enabled')
    DEVICE = torch.device('cpu')

print('Device :', DEVICE)
print("Torch version:{}".format(torch.__version__))
print("cuda version: {}".format(torch.version.cuda))
print("cudnn version:{}".format(torch.backends.cudnn.version()))






