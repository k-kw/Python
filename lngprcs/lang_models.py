import torch
import torch.nn as nn
import torch.nn.functional as F

#lstmモデル
class MyLSTM(nn.Module):
  def __init__(self, vocsize, posn, hdim):
    super(MyLSTM, self).__init__()
    self.embd = nn.Embedding(vocsize, hdim, padding_idx = 0)
    self.lstm = nn.LSTM(hdim, hdim, batch_first = True)
    self.ln = nn.Linear(hdim, posn)
  def forward(self, x):
    x = self.embd(x)
    x, (hn, cn) = self.lstm(x)
    x = self.ln(x)
    return x

#lstm双方向モデル
class MyLSTM_dibi(nn.Module):
  def __init__(self, vocsize, posn, hdim):
    super(MyLSTM_dibi, self).__init__()
    self.embd = nn.Embedding(vocsize, hdim, padding_idx = 0)
    self.lstm = nn.LSTM(hdim, hdim, batch_first = True, num_layers = 2, bidirectional = True)
    self.ln = nn.Linear(hdim * 2, posn)
  def forward(self, x):
    x = self.embd(x)
    x, (hn, cn) = self.lstm(x)
    x = self.ln(x)
    return x



#nmtモデル
class MyNMT(nn.Module):
  def __init__(self, jv, ev, k):
    super(MyNMT, self).__init__()
    self.jemb = nn.Embedding(jv, k)
    self.eemb = nn.Embedding(ev, k)
    self.lstm1 = nn.LSTM(k, k, num_layers = 2)
    self.lstm2 = nn.LSTM(k, k, num_layers = 2)
    self.w = nn.Linear(k, ev)
  def forward(self, jline, eline):
    x = self.jemb(jline)
    ox, (hnx, cnx) = self.lstm1(x)
    y = self.eemb(eline)
    oy, (hny, cny) = self.lstm2(y, (hnx, cnx))
    out = self.w(oy)
    return out

#attention_nmtモデル
class MyAttNMT(nn.Module):
  def __init__(self, jv, ev, k):
    super(MyAttNMT, self).__init__()
    self.jemb = nn.Embedding(jv, k)
    self.eemb = nn.Embedding(ev, k)
    self.lstm1 = nn.LSTM(k, k, num_layers = 2, batch_first = True)
    self.lstm2 = nn.LSTM(k, k, num_layers = 2, batch_first = True)
    self.Wc = nn.Linear(2*k, k)
    self.W = nn.Linear(k, ev)
  def forward(self, jline, eline):
    x = self.jemb(jline)
    ox, (hnx, cnx) = self.lstm1(x)
    y = self.eemb(eline)
    oy, (hny, cny) = self.lstm2(y, (hnx, cnx))
    #内積を計算するため入れ替える
    ox1 = ox.permute(0, 2, 1)
    #行列の積を求めると、各要素が書く中間表現の内積(類似度)になる
    sim = torch.bmm(oy, ox1)
    #softmaxのために変形
    bs, yws,xws = sim.shape
    sim2 = sim.reshape(bs*yws, xws)
    #softmax後元に戻す
    alpha = F.softmax(sim2, dim = 1).reshape(bs, yws, xws)
    ct = torch.bmm(alpha, ox)
    #連結
    oy1 = torch.cat([ct, oy], dim = 2)
    oy2 = self.Wc(oy1)
    return self.W(oy2)


