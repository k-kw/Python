import matplotlib.pyplot as plt
from torch.autograd import Variable
import time
import torch
import numpy as np
import py_func.my_numpy_class as mnc


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

        return tl, vl, ta, va


