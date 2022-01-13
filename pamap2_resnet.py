import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from sklearn.metrics import f1_score
from torch_scatter import scatter_max
from torch_geometric.nn import DenseGCNConv, ChebConv, BatchNorm, PairNorm, GraphNorm
import sklearn.metrics as metrics
from pamap2_tensor_dataset import get_dataset
from tnda_dataset import get_tnda_dataset
from torch_geometric.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

warnings.filterwarnings("ignore")
BOOLEAN_STRANFER = False

def get_data():
    train, test = get_dataset('data_/pamap2.dat')
    return DataLoader(dataset=train, shuffle=True, batch_size=128), DataLoader(dataset=test, shuffle=True,
                                                                              batch_size=256,
                                                                              )


class OwnGCN(nn.Module):
    def __init__(self, device):
        super(OwnGCN, self).__init__()
        self.device = device
        self.conv1 = ChebConv(128, 256, 2)
        self.bn1 = GraphNorm(256)
        self.conv2 = ChebConv(256, 512, 3)
        self.bn2 = GraphNorm(512)
        self.conv3 = ChebConv(512, 256, 3)
        self.bn3 = GraphNorm(256)
        self.conv4 = ChebConv(256, 128, 2)


        self.conv5 = ChebConv(128, 256, 2)
        self.bn5 = GraphNorm(256)
        self.conv6 = ChebConv(256, 512, 3)
        self.bn6 = GraphNorm(512)
        self.conv7 = ChebConv(512, 256, 3)
        self.bn7 = GraphNorm(256)
        self.conv8 = ChebConv(256, 128, 2)


        self.conv9 = ChebConv(128, 256, 2)
        self.bn9 = GraphNorm(256)
        self.conv10 = ChebConv(256, 512, 3)
        self.bn10 = GraphNorm(512)
        self.conv11 = ChebConv(512, 256, 3)
        self.bn11 = GraphNorm(256)
        self.conv12 = ChebConv(256, 128, 2)


        self.conv13 = ChebConv(128, 256, 2)
        self.bn13 = GraphNorm(256)
        self.conv14 = ChebConv(256, 512, 3)
        self.bn14 = GraphNorm(512)
        self.conv15 = ChebConv(512, 256, 3)
        self.bn15 = GraphNorm(256)
        self.conv16 = ChebConv(256, 128, 2)

        self.linear1 = torch.nn.Linear(128, 64)
        self.linear2 = torch.nn.Linear(64, 10)   #10分类



    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1, negative_slope=0.2)
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2, negative_slope=0.2)
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.leaky_relu(x3, negative_slope=0.2)
        x4 = self.conv4(x3, edge_index)
        x4 += x
        x4 = F.relu(x4)


        x5 = self.conv5(x4, edge_index)
        x5 = self.bn5(x5)
        x5 = F.leaky_relu(x5, negative_slope=0.2)
        x6 = self.conv6(x5, edge_index)
        x6 = self.bn6(x6)
        x6 = F.leaky_relu(x6, negative_slope=0.2)
        x7 = self.conv7(x6, edge_index)
        x7 = self.bn7(x7)
        x7 = F.leaky_relu(x7, negative_slope=0.2)
        x8 = self.conv8(x7, edge_index)
        x8 += x
        x8 = F.relu(x8)


        x9 = self.conv9(x8, edge_index)
        x9 = self.bn9(x9)
        x9 = F.leaky_relu(x9, negative_slope=0.2)
        x10 = self.conv10(x9, edge_index)
        x10 = self.bn10(x10)
        x10 = F.leaky_relu(x10, negative_slope=0.2)
        x11 = self.conv11(x10, edge_index)
        x11 = self.bn11(x11)
        x11 = F.leaky_relu(x11, negative_slope=0.2)
        x12 = self.conv12(x11, edge_index)
        x12 += x
        x12 = F.relu(x12)


        x13 = self.conv13(x12, edge_index)
        x13 = self.bn13(x13)
        x13 = F.leaky_relu(x13, negative_slope=0.2)
        x14 = self.conv14(x13, edge_index)
        x14 = self.bn14(x14)
        x14 = F.leaky_relu(x14, negative_slope=0.2)
        x15 = self.conv15(x14, edge_index)
        x15 = self.bn15(x15)
        x15 = F.leaky_relu(x15, negative_slope=0.2)
        x16 = self.conv16(x15, edge_index)
        x16 += x
        x16 = F.relu(x16)
        # x, _ = scatter_max(x, data.batch, dim=0)
        # global_mean_pool 最大池化
        out = pyg_nn.global_mean_pool(x16, data.batch)  # 平均池化
        out = self.linear1(out)
        out = F.tanh(out)
        out = self.linear2(out)

        #out = F.tanh(out)
        #out = self.linear3(out)
        #out = F.tanh(out)
        #out = self.linear4(out)
        return out


def main():
    # # os.environ[]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train_loader, test_loader = get_data()
    # #device = torch.cuda.set_device(1)
    # net = OwnGCN( device=device)
    # net.to(device)
    #
    # params = torch.load("tnda_pam.pth") # 加载参数
    # net.load_state_dict(params, False)
    # print("迁移学习参数加载成功！")

    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_data()
    net = OwnGCN( device=device)
    net.to(device)
    if BOOLEAN_STRANFER:
        print('STRANFER_LEARNING')
        pretrained_dict = torch.load("mmhealth_new_pam.pth")  # 加载参数
        # 过滤操作
        new_dict = {}
        for k, v in pretrained_dict.items():
            if 'linear2' not in k:
                new_dict[k] = v
        state_dict = net.state_dict()
        print(1111)
        state_dict.update(new_dict)


        net.load_state_dict(state_dict=state_dict)
        print("迁移学习参数加载成功！")

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # train
    criterion = nn.CrossEntropyLoss()

    dev_loss = []
    dev_accuracies = []
    train_loss = []




    for epoch in range(200):
        epoch_loss = 0.0
        batch_num = 0
        net.train()
        for batch in train_loader:
            batch_num += 1
            net.zero_grad()
            batch = batch.to(device)
            output = net(batch)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
        print('epoch ' + str(epoch) + ', loss: {:.4f}'.format(epoch_loss / batch_num), end='  ')
        train_loss.append(epoch_loss / batch_num)

        correct = 0
        total = 0
        batch_num = 0
        loss = 0
        f1score = 0
        target = []
        predict = []
        net.eval()
        with torch.no_grad():
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
            acc_num = float(correct / total)
            print('Test Accuracy: {:.2f} %'.format(100 * acc_num), end='  ')
            print('Test Loss: {:.3f}...'.format(loss.cpu().item() / batch_num), end='  ')
            print("F1-Score: {:.4f}...".format(metrics.f1_score(target, predict, average='weighted')), end='  ')
            print('Precision: {:.4f}...'.format(metrics.precision_score(target, predict, average='weighted')), end='  ')
            print('Recall: {:.4f}...'.format(metrics.recall_score(target, predict, average='weighted')))
            dev_accuracies.append(float(correct / total))
            dev_loss.append(loss.cpu().item() / batch_num)
            if acc_num >0.981:
                break

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

    LABELS = ['1','2','3', '4', '5', '6', '7', '8', '9', '10']
    confusion_matrix = metrics.confusion_matrix(target, predict,normalize='true')  ###混淆矩阵TPTF
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=LABELS)
    disp.plot()
    #confusion_matrix = metrics.confusion_matrix(target, predict)  ###混淆矩阵TPTF
    ##plt.figure(figsize=(16, 14))
    #sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    #plt.title("CONFUSION MATRIX_RFC : ")
    #plt.ylabel('True Label')
    #plt.xlabel('Predicted label')
    #plt.savefig('cmatrix.png')
    plt.show()
    print('Test Accuracy: {:.2f} %'.format(100 * float(correct / total)), end='  ')
    print(f'Test Loss: {loss.cpu().item() / batch_num:.3f}')


    print(classification_report(target, predict, digits=4))

    print("原网络及参数:")


    # test
    #for layer in net.layers:
    #    layer.trainable = False
    print("原网络及参数:")

    for idx,m in enumerate(net.children()):
        print(idx,"-",m)

    for p in net.parameters():
        print(type(p.data),p.size())

    #torch.save(net.state_dict(),"all_net_pam.pth") # 保存参数





if __name__ == '__main__':
    main()