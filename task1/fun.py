import torch

import torch.nn.functional as F


def polynomial_fun(w, x):
    """
    Very basic polynomial.
    w - param tensor
    x - input tensor
    """
    return torch.sum(w[i] * x**i for i in range(len(w)))


def polynomial_fun_soft(w, M, x, max_M):
    """
    Polynomial where terms are weighted with softmax of M.
    w - param tensor
    M - degree tensor
    x - input tensor
    max_M - max number of degrees allowed
    """
    # Get a tensor with all possible terms
    degrees = torch.arange(max_M + 1, dtype=torch.float64, device=x.device).unsqueeze(0)
    expanded = M.expand_as(degrees)
    # Weigh each term in accordance to the softmax of M
    soft_weights = F.softmax(expanded * degrees, dim=1)
    # Build powers of x
    x_powers = x.pow(degrees)
    # Get the sum the weighted powers
    y_pred = torch.sum(soft_weights * w.T * x_powers, dim=1)
    return y_pred


def fit_polynomial_ls(x, t, M):
    """
    Fit polynomial solving for minimal least squares.
    """
    N = x.shape[0]
    A = torch.zeros((N, M + 1), dtype=torch.float64)

    for i in range(M + 1):
        A[:, i] = x.pow(i)
    
    result = torch.linalg.lstsq(A, t.unsqueeze(1))
    w = result.solution.squeeze()
    
    return w


def fit_polynomial_sgd(x, t, M, lr, batch_size, epochs=50000, log_step=10000, momentum=0.9):
    """
    Fit polynomial using SGD.
    x - input data
    t - observed values
    M - number of degrees in the poly
    lr - learning rate
    batch_size - batch size
    epochs - number of epochs
    log_step - step between the logs
    momentum - for SGD optimizer
    """
    # Init weights randomly
    w = torch.randn(M + 1, 1, dtype=torch.float64, requires_grad=True)
    
    # Init the optimizer
    optimizer = torch.optim.Adam([w], lr=lr)

    # Use mean squared error as loss
    criterion = torch.nn.MSELoss()

    # Construct the dataset for the convinience of the training loop
    dataset = torch.utils.data.TensorDataset(x.unsqueeze(1), t.unsqueeze(1))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        for batch_x, batch_t in dataloader:
            optimizer.zero_grad()
            y_pred = polynomial_fun(w, batch_x)
            loss = criterion(y_pred, batch_t)
            loss.backward()
            optimizer.step()
        
         # Log only when log_step requires us to
        if epoch % log_step == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {loss.item()}")

    # Return weights
    return w.detach().squeeze()


def fit_polynomial_sgd_a(x, t, max_M, lr, batch_size, epochs=500000, log_step=10000, momentum=0.9):
    """
    Fit polynomial using SGD where M is a learnable parameter.
    We are going to do this by having a fixed number of terms equal to max_M.
    We will weight each term contribution by a softmax of M.
    This will turn off some terms when M is low.
    x - input data
    t - observed values
    max_M - maximum number of degrees to pad to
    lr - learning rate
    batch_size - batch size
    epochs - number of epochs
    log_step - step between the logs
    momentum - for SGD optimizer
    """
    # Init weights randomly
    w = torch.randn(max_M + 1, 1, dtype=torch.float64, requires_grad=True)
    # Init M to maximum allowed value, so that we can have all terms in the beginning
    M = torch.tensor([float(max_M)], requires_grad=True)

    # Init the optimizer
    optimizer = torch.optim.Adam([w, M], lr=lr)

    # Use MSE as loss
    criterion = torch.nn.MSELoss()

    # Construct the dataset for the convinience of the training loop
    dataset = torch.utils.data.TensorDataset(x.unsqueeze(1), t.unsqueeze(1))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        for batch_x, batch_t in dataloader:
            optimizer.zero_grad()

            # Run prediction
            y = polynomial_fun_soft(w, M, batch_x, max_M)
            y = y.unsqueeze(1)

            # Calculate loss
            loss = criterion(y, batch_t)

            # And run the optimizer
            loss.backward()
            optimizer.step()
        
        if epoch % log_step == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {loss.item()} - M: {M.item()}")

    # Return both weights and M
    return w.detach().squeeze(), M.detach().squeeze()


def rmse(predictions, targets):
    """
    RMSE implementation that trims prediction dimensions to targets in case they are uneven.
    """
    if predictions.shape[0] > targets.shape[0]:
        predictions = predictions[:targets.shape[0]]
    mse = torch.mean((predictions - targets) ** 2)
    return torch.sqrt(mse)
