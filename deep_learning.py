import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from math import sqrt
import random

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_series(n=2000):
    t = np.arange(n)
    trend = 0.005*t
    s1 = np.sin(2*np.pi*t/24)
    s2 = np.cos(2*np.pi*t/12)
    s3 = np.sin(2*np.pi*t/48)
    noise = np.random.normal(0,0.15,(n,3))

    x1 = trend + s1 + noise[:,0]
    x2 = 0.5*trend + s2 + noise[:,1]
    x3 = 0.7*x1 + 0.3*x2 + s3 + noise[:,2]

    return np.vstack([x1,x2,x3]).T


def create_windows(data, lookback=40, horizon=5):
    X,y = [],[]
    for i in range(len(data)-lookback-horizon):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+horizon])
    return np.array(X), np.array(y)


def RMSE(a,b):
    return sqrt(mean_squared_error(a,b))


def MASE(y_true, y_pred, train):
    naive = np.mean(np.abs(train[1:] - train[:-1]))
    return np.mean(np.abs(y_true-y_pred))/naive


class BaselineLSTM(nn.Module):
    def __init__(self, input_dim, hidden, out):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden,out)

    def forward(self,x):
        out,_ = self.lstm(x)
        out = self.drop(out[:,-1])
        return self.fc(out)


class BahdanauAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.W1 = nn.Linear(hidden,hidden)
        self.W2 = nn.Linear(hidden,hidden)
        self.V  = nn.Linear(hidden,1)

    def forward(self, hidden, enc_out):
        hidden = hidden.unsqueeze(1)
        score = self.V(torch.tanh(self.W1(enc_out)+self.W2(hidden)))
        weights = torch.softmax(score,dim=1)
        context = torch.sum(weights*enc_out,dim=1)
        return context,weights


class Encoder(nn.Module):
    def __init__(self, inp, hidden):
        super().__init__()
        self.lstm = nn.LSTM(inp,hidden,batch_first=True)

    def forward(self,x):
        out,(h,c)=self.lstm(x)
        return out,h,c


class Decoder(nn.Module):
    def __init__(self, hidden, out):
        super().__init__()
        self.attn = BahdanauAttention(hidden)
        self.lstm = nn.LSTM(hidden+out, hidden, batch_first=True)
        self.fc = nn.Linear(hidden,out)

    def forward(self,x,h,c,enc_out):
        context,w = self.attn(h[-1],enc_out)
        x = torch.cat([x,context.unsqueeze(1)],dim=2)
        out,(h,c)=self.lstm(x,(h,c))
        pred=self.fc(out)
        return pred,h,c,w


class Seq2Seq(nn.Module):
    def __init__(self, inp, hidden, out):
        super().__init__()
        self.encoder=Encoder(inp,hidden)
        self.decoder=Decoder(hidden,out)

    def forward(self,src,trg,tf=0.5):
        enc_out,h,c=self.encoder(src)
        dec_in=trg[:,0:1,:]

        outputs=[]
        attn_all=[]

        for t in range(trg.size(1)):
            out,h,c,w=self.decoder(dec_in,h,c,enc_out)
            outputs.append(out)
            attn_all.append(w)

            if random.random()<tf:
                dec_in=trg[:,t:t+1,:]
            else:
                dec_in=out

        return torch.cat(outputs,1),attn_all


def train(model, loader, val_loader, epochs=30):
    opt = optim.Adam(model.parameters(),lr=0.001)
    loss_fn = nn.MSELoss()
    best=999

    for ep in range(epochs):

        model.train()
        tr_loss=0

        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device)

            opt.zero_grad()
            pred,_=model(xb,yb)
            loss=loss_fn(pred,yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step()

            tr_loss+=loss.item()

        model.eval()
        val_loss=0

        with torch.no_grad():
            for xb,yb in val_loader:
                pred,_=model(xb.to(device),yb.to(device))
                val_loss+=loss_fn(pred,yb.to(device)).item()

        print(f"Epoch {ep+1} | Train {tr_loss:.4f} | Val {val_loss:.4f}")

        if val_loss<best:
            best=val_loss
            torch.save(model.state_dict(),"best.pt")


data=generate_series()

scaler=MinMaxScaler()
data=scaler.fit_transform(data)

X,y=create_windows(data)

n=len(X)
tr=int(0.7*n)
va=int(0.85*n)

train_ds=TensorDataset(torch.tensor(X[:tr]).float(),torch.tensor(y[:tr]).float())
val_ds  =TensorDataset(torch.tensor(X[tr:va]).float(),torch.tensor(y[tr:va]).float())
test_ds =TensorDataset(torch.tensor(X[va:]).float(),torch.tensor(y[va:]).float())

train_loader=DataLoader(train_ds,batch_size=64,shuffle=True)
val_loader  =DataLoader(val_ds,batch_size=64)
test_loader =DataLoader(test_ds,batch_size=64)


baseline=BaselineLSTM(3,64,3).to(device)
opt=optim.Adam(baseline.parameters())

for _ in range(15):
    for xb,yb in train_loader:
        pred=baseline(xb.to(device))
        loss=nn.MSELoss()(pred,yb[:,0,:].to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()


model=Seq2Seq(3,64,3).to(device)
train(model,train_loader,val_loader)

model.load_state_dict(torch.load("best.pt"))


preds=[]
truth=[]

with torch.no_grad():
    for xb,yb in test_loader:
        p,_=model(xb.to(device),yb.to(device),0)
        preds.append(p[:,0,:].cpu().numpy())
        truth.append(yb[:,0,:].numpy())

pred=np.vstack(preds)
true=np.vstack(truth)


base_pred=[]
for xb,yb in test_loader:
    base_pred.append(baseline(xb.to(device)).cpu().detach().numpy())
base_pred=np.vstack(base_pred)


print("\n=========== FINAL RESULTS ===========")
print("Baseline RMSE :",RMSE(true,base_pred))
print("Attention RMSE:",RMSE(true,pred))
print("Baseline MASE :",MASE(true,base_pred,data))
print("Attention MASE:",MASE(true,pred,data))


_,attn=model(torch.tensor(X[va:va+1]).float().to(device),
             torch.tensor(y[va:va+1]).float().to(device),0)

attn=np.array([a.squeeze().cpu().detach().numpy() for a in attn])

plt.imshow(attn,cmap="viridis")
plt.title("Attention Heatmap")
plt.xlabel("Input Time Steps")
plt.ylabel("Forecast Steps")
plt.colorbar()
plt.show()