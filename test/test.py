import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
from itertools import product

import sys
sys.path.append('../src/SamPytorchHelper')
from TorchHelper import TorchHelperClass

# hyper-parameters
parameters = dict(
    lr=[0.01],  # [0.01, 0.001],
    batch=[32], # [64, 128],
    shuffle=[True],
    epochs=[1], #[10, 20],
    momentum=[0.9]
    )


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_Relu = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        pred = self.linear_Relu(x)
        return pred


if __name__ == '__main__':
    train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor())
    test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())

    param_values = [v for v in parameters.values()]
    for id, (lr, batch, shuffle, epochs, momentum ) in enumerate(product(*param_values)):
        print("Current Hyperparams id:", id+1)
        train_dataloader = DataLoader(train_data, batch_size=batch, shuffle=shuffle)
        test_dataloader = DataLoader(test_data, batch_size=batch, shuffle=False)

        net = Network()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

        comment = f' epch={epochs} lr={lr} bch={batch}'
        helper = TorchHelperClass(model=net, loss_function=criterion, optimizer=optimizer, comment=comment)
        helper.train_model(train_dataloader, test_dataloader, epochs, 1000)
        helper.save_model('trained_models')
        print()

    """
    tr_model = Network()
    tr_model.load_state_dict(torch.load('trained_models/model_ep_2_acc_85.2.pth'))
    classes = ["T-shirt/top",
               "Trouser",
               "Pullover",
               "Dress",
               "Coat",
               "Sandal",
               "Shirt",
               "Sneaker",
               "Bag",
               "Ankle boot"]
    x, y = iter(test_dataloader).next()
    x, y = x[60], y[60]
    with torch.no_grad():
        pred = tr_model(x)
        predicted, actual = classes[pred.argmax()], classes[y]
    print(f'predicted: {predicted} -  actual: {actual}')
    """