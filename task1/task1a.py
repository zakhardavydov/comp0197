import torch
import numpy as np

import torch.nn.functional as F

from fun import *


if __name__ == "__main__":

    torch.manual_seed(42)
    
    lr = 0.01
    minibatch_size = 36
    epochs = 100000

    n_train, n_test = 20, 10
    true_w = torch.tensor([1, 2, 3], dtype=torch.float64)

    x_train = torch.linspace(-20, 20, n_train, dtype=torch.float64)
    x_test = torch.linspace(-20, 20, n_test, dtype=torch.float64)

    y_train_true = polynomial_fun(true_w, x_train)
    y_test_true = polynomial_fun(true_w, x_test)

    noise_std = 0.5
    t_train = y_train_true + torch.randn(n_train, dtype=torch.float64) * noise_std
    t_test = y_test_true + torch.randn(n_test, dtype=torch.float64) * noise_std

    diff_a_train = t_train - y_train_true
    mean_diff_a_train = diff_a_train.mean().item()
    std_diff_a_train = diff_a_train.std().item()
    y_rmse = rmse(t_train, y_train_true)

    print("Observered Y vs true Y:")
    print(f"    Mean difference: {mean_diff_a_train}")
    print(f"    Standard deviation: {std_diff_a_train}")
    print(f"    RMSE: {y_rmse}")

    print("-------------------")

    max_M = 4
    
    w_sgd, M = fit_polynomial_sgd_a(x_train, t_train, max_M, lr, minibatch_size, epochs=epochs)

    with torch.no_grad():
        degrees = torch.arange(max_M + 1).double()
        M_soft = F.softmax(M * degrees, dim=0)
        
        train_predicted = polynomial_fun_soft(w_sgd, M_soft, x_train.unsqueeze(1), max_M)
        test_predicted = polynomial_fun_soft(w_sgd, M_soft, x_test.unsqueeze(1), max_M)

    diff_train = train_predicted - y_train_true
    train_mean = diff_train.mean().item()
    train_sd = diff_train.std().item()
    
    diff_test = test_predicted - y_test_true
    test_mean = diff_test.mean().item()
    test_sd = diff_test.std().item()

    print("Train mean difference: ", train_mean)
    print("Train standard deviation: ", train_sd)
    print("Test mean difference: ", test_mean)
    print("Test standard deviation: ", test_sd)
