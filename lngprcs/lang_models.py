import torch
import torch.nn as nn
import torch.nn.functional as F

#lstmモデル
class MyLSTM(nn.Module):
  def __init__(self, vocsize, posn, hdim, lstm_num_layers):
    super(MyLSTM, self).__init__()
    self.embd = nn.Embedding(vocsize, hdim, padding_idx = 0)
    self.lstm = nn.LSTM(hdim, hdim, batch_first = True, num_layers = lstm_num_layers)
    
    #lstmのnum_layersをいくつにしても出力サイズはそのままhdim
    self.ln = nn.Linear(hdim, posn)
  def forward(self, x):
    x = self.embd(x)
    x, (hn, cn) = self.lstm(x)
    x = self.ln(x)
    return x

#lstm双方向モデル
class MyLSTM_bidi(nn.Module):
  def __init__(self, vocsize, posn, hdim, lstm_num_layers):
    super(MyLSTM_bidi, self).__init__()
    self.embd = nn.Embedding(vocsize, hdim, padding_idx = 0)
    self.lstm = nn.LSTM(hdim, hdim, batch_first = True, num_layers = lstm_num_layers, bidirectional = True)

    #bidirectional(双方向)にしたため、lstmの出力がhdim*2になる。よってlnの入力も２倍
    self.ln = nn.Linear(hdim * 2, posn)
  def forward(self, x):
    x = self.embd(x)
    x, (hn, cn) = self.lstm(x)
    x = self.ln(x)
    return x



#推論時のforwardを実装
#nmtモデル
class MyNMT(nn.Module):
    def __init__(self, jv, ev, k, jlay_num, elay_num):
        super(MyNMT, self).__init__()
        
        #nn.Embeddingは(1, words_num)を(1, words_num, k)に変換
        self.jemb = nn.Embedding(jv, k)
        self.eemb = nn.Embedding(ev, k)
        self.lstm1 = nn.LSTM(k, k, num_layers = jlay_num, batch_first = True)
        self.lstm2 = nn.LSTM(k, k, num_layers = elay_num, batch_first = True)
        self.w = nn.Linear(k, ev)
    def forward(self, jline, eline):
        x = self.jemb(jline)
        ox, (hnx, cnx) = self.lstm1(x)
        y = self.eemb(eline)
        oy, (hny, cny) = self.lstm2(y, (hnx, cnx))
        out = self.w(oy)
        return out
    
    #推論時に使うエンコーダ部
    def infer_encode(self, jline):
        x = self.jemb(jline)
        ox, (hn, cn) = self.lstm1(x)
        
        #デコーダ用のインスタンス変数を用意
        self.dcd_ox = ox
        self.dcd_hn = hn
        self.dcd_cn = cn
    
    #推論時に使うデコーダ部
    def infer_decode(self, wids):
        y = self.eemb(wids)
        oy, (self.dcd_hn, self.dcd_cn) = self.lstm2(y, (self.dcd_hn, self.dcd_cn))
        oy = self.w(oy)
        return oy



#attention_nmtモデル
class MyAttNMT(nn.Module):
    def __init__(self, jv, ev, k, jlay_num, elay_num):
        super(MyAttNMT, self).__init__()
        self.jemb = nn.Embedding(jv, k)
        self.eemb = nn.Embedding(ev, k)
        self.lstm1 = nn.LSTM(k, k, num_layers = jlay_num, batch_first = True)
        self.lstm2 = nn.LSTM(k, k, num_layers = elay_num, batch_first = True)
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
    
    
   #推論時に使うエンコーダ部
    def infer_encode(self, jline):
        x = self.jemb(jline)
        ox, (hn, cn) = self.lstm1(x)
        
        #デコーダ用にインスタンス変数を用意
        self.dcd_ox = ox
        self.dcd_hn = hn
        self.dcd_cn = cn
    
    #推論時に使うデコーダ部
    def infer_decode(self, wids):
        y = self.eemb(wids)
        oy, (self.dcd_hn, self.dcd_cn) = self.lstm2(y, (self.dcd_hn, self.dcd_cn))
        ox1 = self.dcd_ox.permute(0, 2, 1)
        sim = torch.bmm(oy, ox1)
        bs, yws, xws = sim.shape
        sim2 = sim.reshape(bs*yws, xws)
        alpha = F.softmax(sim2, dim = 1).reshape(bs, yws, xws)
        ct = torch.bmm(alpha, self.dcd_ox)
        oy1 = torch.cat([ct, oy], dim = 2)
        oy2 = self.Wc(oy1)
        oy3 = self.W(oy2)
        return oy3