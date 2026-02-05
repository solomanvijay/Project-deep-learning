import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def generate_series(n_steps=1500):
    t = np.arange(n_steps)
    trend = 0.01 * t
    s1 = np.sin(2*np.pi*t/24)
    s2 = np.cos(2*np.pi*t/12)
    s3 = np.sin(2*np.pi*t/48)
    noise = np.random.normal(0, 0.2, (n_steps, 3))

    x1 = trend + s1 + noise[:,0]
    x2 = 0.5*trend + s2 + noise[:,1]
    x3 = 0.8*x1 + 0.2*x2 + s3 + noise[:,2]

    return np.vstack([x1,x2,x3]).T


def create_windows(data, lookback=30, horizon=1):
    X, y = [], []
    for i in range(len(data)-lookback-horizon):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+horizon])
    return np.array(X), np.array(y)


def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))


def mase(y_true, y_pred, train):
    naive = np.mean(np.abs(train[1:] - train[:-1]))
    return np.mean(np.abs(y_true-y_pred)) / naive


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, dropout=0.3, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out,_ = self.lstm(x)
        out = out[:,-1,:]
        return self.fc(out)


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.unsqueeze(1)
        score = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(hidden)))
        weights = torch.softmax(score, dim=1)
        context = torch.sum(weights * encoder_outputs, dim=1)
        return context, weights


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        outputs,(h,c) = self.lstm(x)
        return outputs, h, c


class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.attn = BahdanauAttention(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim+output_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell, enc_outputs):
        context, weights = self.attn(hidden[-1], enc_outputs)
        x = torch.cat([x, context.unsqueeze(1)], dim=2)
        out,(h,c) = self.lstm(x,(hidden,cell))
        return self.fc(out), h, c, weights


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, output_dim)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        enc_outputs, h, c = self.encoder(src)
        outputs = []
        dec_input = trg[:,0:1,:]

        for t in range(trg.shape[1]):
            out,h,c,_ = self.decoder(dec_input,h,c,enc_outputs)
            outputs.append(out)
            if np.random.rand() < teacher_forcing_ratio:
                dec_input = trg[:,t:t+1,:]
            else:
                dec_input = out

        return torch.cat(outputs, dim=1)


def train_model(model, train_loader, val_loader, epochs=50):
    model.to(device)

    opt = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3)
    loss_fn = nn.MSELoss()

    best = 1e9
    patience = 7
    wait = 0

    for e in range(epochs):
        model.train()
        train_loss = 0

        for xb,yb in train_loader:
            xb,yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            pred = model(xb,yb)
            loss = loss_fn(pred,yb)

            loss.backward()
            opt.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb = xb.to(device), yb.to(device)
                pred = model(xb,yb,0)
                val_loss += loss_fn(pred,yb).item()

        scheduler.step(val_loss)

        print(f"Epoch {e+1} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best:
            best = val_loss
            wait = 0
            torch.save(model.state_dict(),"best.pt")
        else:
            wait += 1
            if wait > patience:
                break


if __name__ == "__main__":

    data = generate_series()

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    X,y = create_windows(data,30,1)

    n = len(X)
    train_end = int(n*0.7)
    val_end   = int(n*0.85)

    X_train,X_val,X_test = X[:train_end],X[train_end:val_end],X[val_end:]
    y_train,y_val,y_test = y[:train_end],y[train_end:val_end],y[val_end:]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train,dtype=torch.float32),
                      torch.tensor(y_train,dtype=torch.float32)),
        batch_size=32, shuffle=True)

    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val,dtype=torch.float32),
                      torch.tensor(y_val,dtype=torch.float32)),
        batch_size=32)

    model = Seq2Seq(3,64,3)
    train_model(model,train_loader,val_loader)

    model.load_state_dict(torch.load("best.pt"))
    model.eval()

    with torch.no_grad():
        X_test_t = torch.tensor(X_test,dtype=torch.float32).to(device)
        y_test_t = torch.tensor(y_test,dtype=torch.float32).to(device)
        pred = model(X_test_t,y_test_t,0).cpu().numpy()

    score_rmse = rmse(y_test[:,0,:],pred[:,0,:])
    print("\nFinal RMSE:", score_rmse)