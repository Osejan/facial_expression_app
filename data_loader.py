# data_loader.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir='fer2013', batch_size=64):

    transform_train = transforms.Compose([
    transforms.Grayscale(),  # FER2013 is grayscale
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

    transform_test = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = datasets.ImageFolder("fer2013/train", transform=transform_train)
    test_data = datasets.ImageFolder("fer2013/test", transform=transform_test)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    class_names = train_data.classes

    return train_loader, test_loader, train_data.classes  # return label names too
