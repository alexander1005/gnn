import torch
import torch.nn.functional as F
import torch.nn as nn
import json
import torch_geometric.nn as pyg_nn
from sklearn.metrics import f1_score
from torch_scatter import scatter_max
from torch_geometric.nn import DenseGCNConv, ChebConv, BatchNorm, PairNorm, GraphNorm
import sklearn.metrics as metrics
from mmhealth_dataset import get_dataset
from torch_geometric.data import DataLoader
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

warnings.filterwarnings("ignore")


def get_data():
    train, test = get_dataset('data_/mealth.dat')
    return DataLoader(dataset=train, shuffle=True, batch_size=64), DataLoader(dataset=test, shuffle=True,
                                                                              batch_size=128,
                                                                              )


class OwnGCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c, device):
        super(OwnGCN, self).__init__()
        self.device = device
        self.conv1 = ChebConv(128, 256, 4)
        self.bn1 = GraphNorm(256)
        self.conv2 = ChebConv(256, 512, 4)
        self.bn2 = GraphNorm(512)
        self.conv3 = ChebConv(512, 256, 4)
        self.bn3 = GraphNorm(256)
        self.conv4 = ChebConv(256, 128, 4)
        self.bn4 = GraphNorm(128)

        self.linear1 = torch.nn.Linear(128, 64)
        self.linear2 = torch.nn.Linear(64, 12)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1, negative_slope=0.1)
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2, negative_slope=0.1)
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.leaky_relu(x3, negative_slope=0.1)
        x4 = self.conv4(x3, edge_index)
        x4 = self.bn4(x4)
        x4 += x
        x4 = F.relu(x4)
        # x, _ = scatter_max(x, data.batch, dim=0)
        # global_mean_pool 最大池化
        out = pyg_nn.global_mean_pool(x4, data.batch)  # 平均池化
        out = self.linear1(out)
        out = F.tanh(out)
        out = self.linear2(out)
        return out


def main():
    # os.environ[]
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.cuda.set_device(1)
    train_loader, test_loader = get_data()
    # loader_test, dataset_test = get_data(mode='test')
    net = OwnGCN(in_c=24, hid_c=200, out_c=5, device=device)
    net.to(device)
    #params = torch.load("net_pam.pth") # 加载参数
    #net.load_state_dict(params, False)
    # data = cora[0].to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # train
    criterion = nn.CrossEntropyLoss()
    net.train()

    dev_loss = []
    dev_accuracies = []
    train_loss = []


    for epoch in range(100):
        epoch_loss = 0.0
        batch_num = 0
        for batch in train_loader:
            batch_num += 1
            net.zero_grad()
            batch = batch.to(device)
            output = net(batch)
            loss = criterion(output, batch.y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'epoch {epoch}, loss: {epoch_loss/batch_num}')
        train_loss.append(epoch_loss/batch_num)

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
        print(f'Test Loss: {loss.cpu().item() / batch_num:.3f}', end='    ')
        print("F1-Score: {:.4f}...".format(metrics.f1_score(target, predict, average='weighted')))
        print('Precision', metrics.precision_score(target, predict, average='weighted'))
        print('Recall', metrics.recall_score(target, predict, average='weighted'))
        dev_accuracies.append(float(correct / total))
        dev_loss.append(loss.cpu().item() / batch_num)

        # print('F1-score:', metrics.f1_score(y_true, y_pred))
    plt.figure(figsize=(12, 8))

    plt.plot(np.array(train_loss), "r--", label="Training Loss")

    plt.plot(np.array(dev_loss), "r-", label="Test Loss")
    plt.plot(np.array(dev_accuracies), "g-", label="Test Accuracy")

    plt.title("Training session's progress over iterations")
    plt.legend(loc='upper right', shadow=True)
    plt.ylabel('Training progress(Loss or accuracy)')
    plt.xlabel('Training EPOCH')
    plt.ylim(0)
    plt.savefig('Training iterations.png')
    plt.show()
    filename = "mhealth_renet_trainloss.txt"
    with open(filename, 'w') as f_obj2:
        json.dump(train_loss, f_obj2)
    filename = "mhealth_renet_devloss.txt"
    with open(filename, 'w') as f_obj:
        json.dump(dev_loss, f_obj)
    filename = "mhealth_resnet_devacc.txt"
    with open(filename, 'w') as f_obj1:
        json.dump(dev_accuracies, f_obj1)

    # test
    net.eval()

    correct = 0
    total = 0
    batch_num = 0
    loss = 0
    y_valid = []
    y_pred  = []
    target = []
    predict = []
    for data in test_loader:
        data = data.to(device)
        outputs = net(data)
        loss += criterion(outputs, data.y)
        _, predicted = torch.max(outputs, 1)

        total += data.y.size(0)
        batch_num += 1
        correct += (predicted == data.y).sum().cpu().item()
        predict.extend(predicted.detach().cpu().numpy())
        target.extend(data.y.detach().cpu().numpy())
    LABELS = ['1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    confusion_matrix = metrics.confusion_matrix(target, predict)  ###混淆矩阵TPTF
    plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("CONFUSION MATRIX_RFC : ")
    plt.ylabel('True Label')
    plt.xlabel('Predicted label')
    #plt.savefig('cmatrix.png')
    plt.show()
    print('Test Accuracy: {:.2f} %'.format(100 * float(correct / total)), end='  ')
    print(f'Test Loss: {loss.cpu().item() / batch_num:.3f}')


if __name__ == '__main__':
    main()

