import torchvision

train_set = torchvision.datasets.CIFAR10(root="./dataset" , train = True , download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train= False , download=True)

print(train_set[0])
