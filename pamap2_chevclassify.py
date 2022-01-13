import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from sklearn.metrics import f1_score
from torch_scatter import scatter_max
from torch_geometric.nn import DenseGCNConv, ChebConv, BatchNorm, PairNorm, GraphNorm
import sklearn.metrics as metrics
from pamap2_dataset import get_dataset
from torch_geometric.data import DataLoader
import warnings

warnings.filterwarnings("ignore")


def get_data():
    train, test = get_dataset('data_/pamap2.dat')
    return DataLoader(dataset=train, shuffle=True, batch_size=64), DataLoader(dataset=test, shuffle=True,
                                                                              batch_size=128,
                                                                              )


class OwnGCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c, device):
        super(OwnGCN, self).__init__()
        self.device = device
        self.conv1 = ChebConv(128, 190, 2)
        self.bn1 = GraphNorm(190)

        self.conv2 = ChebConv(190, 256, 2)
        self.bn2 = GraphNorm(256)
        self.conv3 = ChebConv(256, 169, 2)
        self.bn3 = GraphNorm(169)
        self.conv4 = ChebConv(169, 190, 5)

        self.conv5 = ChebConv(190, 256, 1)
        self.conv6 = ChebConv(256, 128, 3)
        self.bn6 = GraphNorm(128)
        self.linear1 = torch.nn.Linear(128, 64)
        self.linear2 = torch.nn.Linear(64, 10)   #10分类

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = self.conv6(x, edge_index)
        x = self.bn6(x)
        # x = self.bn(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        # x, _ = scatter_max(x, data.batch, dim=0)
        # global_mean_pool 最大池化
        x = pyg_nn.global_mean_pool(x, data.batch)  # 平均池化
        x = self.linear1(x)
        x = F.tanh(x)
        x = self.linear2(x)
        return x


def main():
    # os.environ[]
    #
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_data()
    # loader_test, dataset_test = get_data(mode='test')
    device = torch.cuda.set_device(1)
    net = OwnGCN(in_c=24, hid_c=200, out_c=5, device=device)
    net.to(device)
    # data = cora[0].to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # train
    criterion = nn.CrossEntropyLoss()
    net.train()

    for epoch in range(50):
        epoch_loss = 0.0
        for batch in train_loader:
            net.zero_grad()
            batch = batch.to(device)
            output = net(batch)
            loss = criterion(output, batch.y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'epoch {epoch}, loss: {epoch_loss}')

        correct = 0
        total = 0
        batch_num = 0
        loss = 0
        f1score = 0
        target = []
        predict = []
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        for data in test_loader:
            data = data.to(device)
            outputs = net(data)
            loss += criterion(outputs, data.y)
            _, predicted = torch.max(outputs, 1)
            total += data.y.size(0)
            batch_num += 1
            correct += (predicted == data.y).sum().cpu().item()
            # top_p, top_class = outputs.topk(1, dim=1)
            # equals = top_class == data.y.view(*top_class.shape).long()
            # accuracy += torch.mean(ex/quals.type(torch.FloatTensor))
            # f1score += metrics.f1_score(top_class.cpu(), data.y.view(*top_class.shape).long().cpu(),
            #                             average='weighted')
            predict.extend(predicted.detach().cpu().numpy())
            target.extend(data.y.detach().cpu().numpy())
        print('Test Accuracy: {:.2f} %'.format(100 * float(correct / total)), end='  ')
        print(f'Test Loss: {loss.cpu().item() / batch_num:.3f}', end='  ')
        print("F1-Score: {:.4f}...".format(metrics.f1_score(target, predict, average='weighted')), end='  ')
        print('Precision: {:.4f}...'.format(metrics.precision_score(target, predict, average='weighted')), end='  ')
        print('Recall: {:.4f}...'.format(metrics.recall_score(target, predict, average='weighted')))

    # test
    net.eval()

    correct = 0
    total = 0
    batch_num = 0
    loss = 0
    for data in test_loader:
        data = data.to(device)
        outputs = net(data)
        loss += criterion(outputs, data.y)
        _, predicted = torch.max(outputs, 1)
        total += data.y.size(0)
        batch_num += 1
        correct += (predicted == data.y).sum().cpu().item()
    print('Test Accuracy: {:.2f} %'.format(100 * float(correct / total)), end='  ')
    print(f'Test Loss: {loss.cpu().item() / batch_num:.3f}')


if __name__ == '__main__':
    main()
