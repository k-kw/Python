import matplotlib.pyplot as plt
from torch.autograd import Variable
import time
import torch
from torch.utils.data import Dataset
import numpy as np

#深層学習前の標準化
def tensor_norm_DL(tensor, mean = None, std = None):
    if mean == None:
        mean = torch.mean(tensor)
    if std == None:
        std = torch.std(tensor)
    re_tensor = (tensor - mean)/std
    return re_tensor, mean, std

#正規化
def tensor_norm(tensor, max = None, min = None):
    if max == None:
        max = torch.max(tensor)
    if min == None:
        min = torch.min(tensor)
    re_tensor = (tensor - min)/(max - min)
    return re_tensor, max, min

#numpy配列から自力でデータセットを作成するクラス
class My_dataset:
    def __init__(self, numpydata, numpylabel):
        self.data = numpydata
        self.labels = numpylabel
        self.cftest = False
        self.length = numpydata.shape[0]
    
    #numpyからテンソル
    def numpy2tensor(self):
        self.data = torch.tensor(self.data, dtype = torch.float32)
        self.labels = torch.tensor(self.labels, dtype = torch.float32)
    
    #numpyからテンソル、ラベルはint64で
    def numpy2tensor_labelint(self):
        self.data = torch.tensor(self.data, dtype = torch.float32)
        self.labels = self.labels.astype(int)
        self.labels = torch.tensor(self.labels, dtype = torch.int64)




    #-------------------データを順番は変えずに分割-------------------
    def splitdata(self, lentrain, lenval = None):
        if lenval == None:
            #訓練データと評価データに分割
            self.data_t, self.data_v = self.data[:lentrain], \
                self.data[lentrain:]
            self.label_t, self.label_v = self.labels[:lentrain], \
                self.labels[lentrain:]
        else:
            #訓練、評価、テストに分割
            self.cftest = True
            self.data_t, self.data_v, self.data_test = self.data[:lentrain], \
                self.data[lentrain:lentrain + lenval], self.data[lentrain + lenval:]
            self.label_t, self.label_v, self.label_test = self.labels[:lentrain], \
                self.labels[lentrain:lentrain + lenval], self.labels[lentrain + lenval:]
    
    



    #---------------------ランダムに訓練、評価、テストに分ける場合---------------------------------

    def tensor_shuffle_split(self, shflidsavepath, trainlen, vallen, shuffleindex=None):
        if shuffleindex==None:
            #shuffleされたindexを受け取っていないときはここで作ってpathに保存
            index = torch.randperm(self.length)
            index = index.to('cpu').detach().numpy().copy()
            np.save(arr = index, file = shflidsavepath)
        else:
            index=shuffleindex
        
        datatrain=[]
        labeltrain=[]
        dataval=[]
        labelval=[]
        datatest=[]
        labeltest=[]


        for cnt, ind in enumerate(index):
            if(cnt+1<=trainlen):
                datatrain.append(self.data[ind])
                labeltrain.append(self.labels[ind])
            elif(cnt+1<=trainlen+vallen):
                dataval.append(self.data[ind])
                labelval.append(self.labels[ind])
            else:
                datatest.append(self.data[ind])
                labeltest.append(self.labels[ind])
        
        self.data_t = torch.stack(datatrain, dim=0)
        self.data_val = torch.stack(dataval, dim=0)
        self.data_test = torch.stack(datatest, dim=0)
        
        self.lebel_t = torch.stack(labeltrain, dim=0)
        self.label_val = torch.stack(labelval, dim=0)
        self.label_test = torch.stack(labeltest, dim=0)
    






    #データセットを訓練データの平均と標準偏差で標準化
    def datanormalize(self):
        self.data_t, self.mean, self.std = tensor_norm_DL(self.data_t)
        self.data_v, _, _ = tensor_norm_DL(self.data_v, self.mean, self.std)
        if self.cftest:
            self.data_test, _, _ = tensor_norm_DL(self.data_test, self.mean, self.std)
    
    #ラベルが正解画像の時、ラベルをmaxmin正規化
    def labelnormalize(self):
        self.label_t, max, min = tensor_norm(self.label_t)
        self.label_v, _, _ = tensor_norm(self.label_v, max, min)
        if self.cftest:
            self.label_test, _, _ = tensor_norm(self.label_test, max, min)
    
    #テンソルからデータセットに
    def tensor2dataset(self):
        self.dataset_train = torch.utils.data.TensorDataset(self.data_t, self.label_t)
        self.dataset_val = torch.utils.data.TensorDataset(self.data_v, self.label_v)
        if self.cftest:
            self.dataset_test = torch.utils.data.TensorDataset(self.data_test, self.label_test)






def channeltensor_mean_std(tensors):
    """
    tensors's shape must be Length,C,...
    """

    mean, std = [], []

    for channel in range(tensors.shape[1]):
        mean.append(torch.mean(tensors[:, channel, :]))
        std.append(torch.std(tensors[:, channel, :]))

    return mean, std



#transformsのNormalizeが２次元データセットのみなのでこのクラスは使えない
#torchのデータセットクラスを継承してnumpy配列からデータセットを作成
class classifydataset(Dataset):
    """
    data's shape must be Length,C,W
    """
    def __init__(self, data, labels, transform = None):
        #self.super().__init__()
        self.data = torch.tensor(data, dtype = torch.float32)
        tmplabels = labels.astype(int)
        self.labels = torch.tensor(tmplabels, dtype = torch.int64)
        self.transform = transform
    
    def datameanstd(self):
        """
        return mean-list and std-list of data
        """
        return channeltensor_mean_std(self.data)
    
    def settransform(self, transform):
        self.transform = transform


    def __getitem__(self, idx):
        outdata = self.data[idx]
        outlabel = self.labels[idx]

        if self.transform:
            outdata = self.transform(outdata)
        
        return outdata, outlabel
    
    def __len__(self):
        return self.data.shape[0]


class decodedataset(Dataset):
    """
    data's shape must be Length,C,W
    origimgs's shape must be Length,C,H,W
    """
    def __init__(self, data, origimgs, transform1 = None, transform2 = None):
        #self.super().__init__()
        self.data = torch.tensor(data, dtype = torch.float32)
        self.origimgs = torch.tensor(origimgs, dtype = torch.float32)
        
        self.transform1 = transform1
        self.transform2 = transform2

    def datameanstd(self):
        """
        return mean-list and std-list of data
        """
        return channeltensor_mean_std(self.data)

    def settransform(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2




    def __getitem__(self, idx):
        outdata = self.data[idx]
        outorig = self.origimgs[idx]

        if self.transform1:
            outdata = self.transform1(outdata)
        
        if self.transform2:
            outorig = self.transform2(outorig)
        
        return outdata, outorig
    
    def __len__(self):
        return self.data.shape[0]






        

#正則化関数
#L1正則化
def L1norm(model, alpha, loss):
  l1 = torch.tensor(0., requires_grad=True)
  for w in model.parameters():
    #重みの大きさの総和
    l1 = l1 + torch.norm(w, 1)
  loss = loss + alpha*l1
  return loss

#L2正則化
def L2norm(model, lamda, loss):
  l2 = torch.tensor(0., requires_grad=True)
  for w in model.parameters():
    #重みの大きさの2乗和
    l2 = l2 + torch.norm(w)**2
  loss = loss + lamda*l2/2
  return loss


class My_train:
    
    def __init__(self, model, device, lossf, optimizer, train_loader, val_loader):
        self.model = model
        self.device = device
        self.lossf = lossf
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        #学習曲線
        self.tl = None
        self.vl = None
        self.ta = None
        self.va = None
    
    def val(self, dataloader):
        self.model.eval()
        correct = 0
        val_loss = 0
        
        predict_list = []
        label_list = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs, labels = Variable(inputs), Variable(labels)
                outputs = self.model(inputs)
                loss = self.lossf(outputs, labels)

                val_loss += loss.item()

                predicted = torch.argmax(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                for p in predicted:
                    p = int(p)
                    predict_list.append(p)
                
                for l in labels:
                    l = int(l)
                    label_list.append(l)

            acc = float(correct)/len(dataloader.dataset)
            val_loss = val_loss/len(dataloader.dataset)
        
        return acc, val_loss, predict_list, label_list


    def train(self, epochs, L1 = False, alpha = None, L2 = False, lamda = None):
        t1 = time.time()
        tl = []
        vl = []
        ta = []
        va = []

        for epoch in range(epochs):
            self.model.train()

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs, labels = Variable(inputs), Variable(labels)
                outputs = self.model(inputs)
                loss = self.lossf(outputs, labels)
            

                #正則化(weight decay)
                if L2:
                    loss = L2norm(self.model, lamda, loss)
                elif L1:
                    loss = L1norm(self.model, alpha, loss)
                
                #勾配初期化、逆伝搬で勾配求める、重み更新
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            

            vacc, vloss, _, _ = self.val(self.val_loader)
            tacc, tloss, _, _ = self.val(self.train_loader)

            tl.append(tloss)
            vl.append(vloss)
            ta.append(tacc)
            va.append(vacc)

            print(f'-----------------------epoch{epoch+1}------------------------------')
            print(f'val_acc{vacc} ,train_acc{tacc}')
            t2=time.time()
            caltime=(t2-t1)/60
            print(f'epochtime:{caltime}分')
            t1=time.time()
        
        self.tl = tl
        self.vl = vl
        self.ta = ta
        self.va = va

        return tl, vl, ta, va
    




    def learning_curv(self, fig_w, fig_h, labelfontsize, scalefontsize):
        rcparams_dic = {
            'figure.figsize': (fig_w,fig_h),
            'axes.labelsize': labelfontsize,
            'xtick.labelsize': scalefontsize,
            'ytick.labelsize': scalefontsize,
            }
        plt.rcParams.update(rcparams_dic)

        #正解率の配列を持っている場合
        if (self.ta != None) or (self.va != None):
            plt.subplots_adjust(wspace=0.4, hspace=0.6)
            plt.subplot(1,2,1)
        
        if self.tl != None:
            plt.plot(range(1, 1 + len(self.tl)), self.tl, label="training")
        plt.plot(range(1, 1 + len(self.vl)), self.vl, label="validation")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        if (self.va != None) or (self.ta != None):
            plt.subplot(1,2,2)
            if self.ta != None:
                plt.plot(range(1, 1 + len(self.ta)), self.ta, label="training")
            plt.plot(range(1, 1 + len(self.va)), self.va, label="validation")
            plt.xlabel('Epochs')
            plt.ylabel('Acc')
            plt.legend()

        plt.show()
    

    def cfmat_save(self, matrix_save_path, vmax, figsy, figsx, fontsize):
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        _, ax = plt.subplots(figsize = (figsy, figsx))

        _, _, predictions, labels = self.val(self.val_loader)

        cm = confusion_matrix(labels,predictions)
        sns.heatmap(cm,square=True,cmap='Blues',annot=True,fmt='d',ax=ax,vmax=vmax,vmin=0,annot_kws={"size":fontsize})
        
        plt.savefig(matrix_save_path)


