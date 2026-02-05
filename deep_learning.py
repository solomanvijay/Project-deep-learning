import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

torch.manual_seed(42)
np.random.seed(42)


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

    data = np.vstack([x1,x2,x3]).T
    return data


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
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
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
        self.V = nn.Linear(hidden_dim,1)

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
        self.lstm = nn.LSTM(hidden_dim + output_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell, enc_outputs):
        context, weights = self.attn(hidden[-1], enc_outputs)
        x = torch.cat([x, context.unsqueeze(1)], dim=2)
        out,(h,c) = self.lstm(x,(hidden,cell))
        pred = self.fc(out)
        return pred, h, c, weights


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, output_dim)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        enc_outputs, h, c = self.encoder(src)
        outputs = []
        attn_list = []
        dec_input = trg[:,0:1,:]

        for t in range(trg.shape[1]):
            out,h,c,weights = self.decoder(dec_input,h,c,enc_outputs)
            outputs.append(out)
            attn_list.append(weights)

            # Teacher forcing
            if np.random.rand() < teacher_forcing_ratio:
                dec_input = trg[:,t:t+1,:]
            else:
                dec_input = out

        outputs = torch.cat(outputs,dim=1)
        return outputs, attn_list


def train_model(model, X_train, y_train, epochs=20):
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for e in range(epochs):
        model.train()
        opt.zero_grad()
        out,_ = model(X_train, y_train)
        loss = loss_fn(out, y_train)
        loss.backward()
        opt.step()
        print(f"Epoch {e+1} Loss: {loss.item():.4f}")


if __name__ == "__main__":

    data = generate_series()

    lookback = 30
    horizon = 1

    X,y = create_windows(data, lookback, horizon)

    split = int(len(X)*0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = torch.tensor(X_train,dtype=torch.float32)
    y_train = torch.tensor(y_train,dtype=torch.float32)
    X_test = torch.tensor(X_test,dtype=torch.float32)
    y_test = torch.tensor(y_test,dtype=torch.float32)


    print("\nTraining Baseline LSTM")

    baseline = LSTMModel(3,64,3)
    opt = optim.Adam(baseline.parameters())

    for _ in range(20):
        opt.zero_grad()
        pred = baseline(X_train)
        loss = nn.MSELoss()(pred,y_train[:,0,:])
        loss.backward()
        opt.step()

    base_pred = baseline(X_test).detach().numpy()


    print("\nTraining Seq2Seq + Attention")

    model = Seq2Seq(3,64,3)
    train_model(model,X_train,y_train,epochs=20)

    model.eval()

    with torch.no_grad():
        pred, attn = model(X_test,y_test,teacher_forcing_ratio=0.0)  # inference mode

    pred = pred.numpy()
    y_test_np = y_test.numpy()


    base_rmse = rmse(y_test_np[:,0,:], base_pred)
    att_rmse = rmse(y_test_np[:,0,:], pred[:,0,:])

    base_mase = mase(y_test_np[:,0,:], base_pred, data)
    att_mase = mase(y_test_np[:,0,:], pred[:,0,:], data)

    print("\n==================== RESULTS ====================")
    print(f"LSTM RMSE : {base_rmse:.4f}")
    print(f"ATTN RMSE : {att_rmse:.4f}")
    print(f"LSTM MASE : {base_mase:.4f}")
    print(f"ATTN MASE : {att_mase:.4f}")


    weights = attn[0][0].squeeze().numpy()

    plt.figure(figsize=(10,4))
    plt.plot(weights)
    plt.title("Attention Weights Across Time Steps")
    plt.xlabel("Time step")
    plt.ylabel("Importance")
    plt.show()
