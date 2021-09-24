import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision


class TorchHelperClass:
    def __init__(self, model, loss_function, optimizer, comment=''):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Network
        self.model = model.to(self.device)

        # loss and optimizer functions
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.epochs = 0

        # Tensorboard
        self.writer = SummaryWriter(comment=comment)
        self.comment = comment

    def train_model(self, train_dataloader, val_dataloader, num_epoch=50, iter_print=100):
        """
        :param train_dataloader: training set
        :param val_dataloader: validation set
        :param num_epoch: the total number of epochs. default = 50
        :param iter_print: indicate when to print the loss after how many iteration. default = 100
        :return: current trained model
        """
        self.epochs = num_epoch
        print(f'------ Training Initiated - device: {self.device}----')
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}")
            self._train(train_dataloader, self.model, self.loss_function, self.optimizer, iter_print, epoch)
            self._test(val_dataloader, self.model, self.loss_function, epoch)
        print("------- Training finished ----------------------------")
        self.writer.close()
        return self.model

    def _train(self, dataloader, model, loss_fn, optimizer, iter_print, epoch):
        total_loss = 0
        total_correct = 0
        size = len(dataloader.dataset)
        batch_size = len(dataloader)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            total_loss += loss.item()
            total_correct += self._get_num_correct(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % iter_print == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"  [iter {current:>5d}/{size:>5d}] --> Loss: {loss:>7f} ")
        self.writer.add_scalar("Training loss per epoch", total_loss/batch_size, epoch+1)
        self.writer.add_scalar("Training accuracy per epoch", total_correct / size, epoch+1)
        self.writer.close()

    def _get_num_correct(self, pred, labels):
        return pred.argmax(dim=1).eq(labels).sum().item()

    def _test(self, dataloader, model, loss_fn, epoch):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += self._get_num_correct(pred, y)
        test_loss /= num_batches
        correct /= size
        print(f" Validation : \n"
              f"  [epoch {epoch+1}/{self.epochs}]--> Accuracy: {(100 * correct):>0.1f}%, --> Avg loss: {test_loss:>8f}")
        self.writer.add_scalar("Validation accuracy per epoch", correct, epoch+1)
        self.writer.add_scalar("Validation loss per epoch", test_loss, epoch+1)

    def save_model(self, path):
        """
        :param path: folder where to save the model
        :return: None
        """
        if path[-1] != '/':
            path = path + '/'
        name = f'{path}model' + self.comment + '.pth'
        torch.save(self.model.state_dict(), name)
        print(f"model '{name}' saved successfully")


