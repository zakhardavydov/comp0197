import torch

import torch.nn.functional as F


def polynomial_fun(w, x):
    return sum(w[i] * x**i for i in range(len(w)))


def polynomial_fun_soft(w, M, x, N):
    degrees = torch.arange(N + 1, dtype=torch.float64, device=x.device).unsqueeze(0)
    M_expanded = M.expand_as(degrees)
    soft_weights = F.softmax(M_expanded * degrees, dim=1)
    x_powers = x.pow(degrees)
    y_pred = torch.sum(soft_weights * w.T * x_powers, dim=1)
    return y_pred


def fit_polynomial_ls(x, t, M):
    N = x.shape[0]
    A = torch.zeros((N, M + 1), dtype=torch.float64)

    for i in range(M+1):
        A[:, i] = x.pow(i)
    
    result = torch.linalg.lstsq(A, t.unsqueeze(1))
    w = result.solution.squeeze()
    
    return w


def fit_polynomial_sgd(x, t, M, lr, minibatch_size, epochs=50000, log_step=10000):
    w = torch.randn(M + 1, 1, dtype=torch.float64, requires_grad=True)
    
    optimizer = torch.optim.Adam([w], lr=lr)

    criterion = torch.nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(x.unsqueeze(1), t.unsqueeze(1))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        for batch_x, batch_t in dataloader:
            optimizer.zero_grad()
            y_pred = polynomial_fun(w, batch_x)
            loss = criterion(y_pred, batch_t)
            loss.backward()
            optimizer.step()
        
        if epoch % log_step == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {loss.item()}")

    return w.detach().squeeze()


def fit_polynomial_sgd_a(x, t, max_M, lr, minibatch_size, epochs=500000, log_step=10000):
    w = torch.randn(max_M + 1, 1, dtype=torch.float64, requires_grad=True)
    M = torch.tensor([5.0], requires_grad=True)

    optimizer = torch.optim.Adam([w, M], lr=lr)
    criterion = torch.nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(x.unsqueeze(1), t.unsqueeze(1))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        for batch_x, batch_t in dataloader:
            optimizer.zero_grad()

            degrees = torch.arange(max_M + 1).double()
            M_soft = F.softmax(M * degrees, dim=0)

            y_pred = polynomial_fun_soft(w, M_soft, batch_x, max_M)
            y_pred = y_pred.unsqueeze(1)

            loss = criterion(y_pred, batch_t)

            loss.backward()
            optimizer.step()
        
        if epoch % log_step == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {loss.item()} - M: {M.item()}")

    return w.detach().squeeze(), M.detach().squeeze()



def rmse(predictions, targets):
    if predictions.shape[0] > targets.shape[0]:
        predictions = predictions[:targets.shape[0]]
    mse = torch.mean((predictions - targets) ** 2)
    return torch.sqrt(mse)
