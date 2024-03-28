import time

import torch
import numpy as np

import torch.nn.functional as F

from fun import *


if __name__ == "__main__":

    torch.manual_seed(42)
    
    lr = 0.01
    minibatch_size = 36

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

    degrees = [2, 3, 4]

    w_estimates = {}
    y_pred_train = {}
    y_pred_test = {}

    mean_diff_b_train = {}
    std_diff_b_train = {}

    sgd_w_estimates = {}
    y_sgd_pred_train = {}
    y_sgd_pred_test = {}

    mean_diff_sgd_train = {}
    std_diff_sgd_train = {}
    mean_diff_sgd_test = {}
    std_diff_sgd_test = {}
    
    poly_time = {}
    sgd_time = {}
    
    y_poly_rmse = {}
    y_sgd_rmse = {}

    w_poly_rmse = {}
    w_sgd_rmse = {}
    
    for degree in degrees:
        print("Fitting poly for degree: ", degree)
        t0 = time.time()
        w_hat = fit_polynomial_ls(x_train, t_train, degree)
        poly_time[degree] = time.time() - t0

        w_poly_rmse[degree] = rmse(w_hat, true_w).item()
        w_estimates[degree] = w_hat
        predicted = polynomial_fun(w_hat, x_train)
        y_pred_train[degree] = predicted
        y_pred_test[degree] = polynomial_fun(w_hat, x_test)

        y_poly_rmse[degree] = rmse(y_pred_test[degree], y_test_true).item()

        diff = predicted - y_train_true
        mean_diff_b_train[degree] = diff.mean().item()
        std_diff_b_train[degree] = diff.std().item()

        print("-------------------")
    
    for degree in degrees:
        print("Fitting SGD for degree: ", degree)
        t0 = time.time()
        w_sgd = fit_polynomial_sgd(x_train, t_train, degree, lr, minibatch_size)
        with torch.no_grad():
            sgd_time[degree] = time.time() - t0
            sgd_w_estimates[degree] = w_sgd
            w_sgd_rmse[degree] = rmse(w_sgd, true_w).item()
            
            y_sgd_pred_train[degree] = polynomial_fun(w_sgd, x_train)
            y_sgd_pred_test[degree] = polynomial_fun(w_sgd, x_test)
            y_sgd_rmse[degree] = rmse(y_sgd_pred_test[degree], y_test_true).item()

            diff_train = y_sgd_pred_train[degree] - y_train_true
            mean_diff_sgd_train[degree] = diff_train.mean().item()
            std_diff_sgd_train[degree] = diff_train.std().item()
            
            diff_test = y_sgd_pred_test[degree] - y_test_true
            mean_diff_sgd_test[degree] = diff_test.mean().item()
            std_diff_sgd_test[degree] = diff_test.std().item()

        print("-------------------")

    print("Comparison between Poly and SGD:")
    
    for degree in degrees:
        print(f"Metrics for degree: {degree}")
        metrics = [
            ["Metric", "Poly", "SGD"],
        ]
        metrics.append(["Time taken", poly_time[degree], sgd_time[degree]])
        metrics.append(["Error mean", mean_diff_b_train[degree], mean_diff_sgd_test[degree]])
        metrics.append(["Error SD", std_diff_b_train[degree], std_diff_sgd_test[degree]])
        metrics.append(["W RMSE", w_poly_rmse[degree], w_sgd_rmse[degree]])
        metrics.append(["Y RMSE", y_poly_rmse[degree], y_sgd_rmse[degree]])

        num_width = max(
            max(len(f"{x:.4g}") for x in poly_time.values()),
            max(len(f"{x:.4g}") for x in sgd_time.values()),
            max(len(f"{x:.4g}") for x in mean_diff_b_train.values()),
            max(len(f"{x:.4g}") for x in mean_diff_sgd_test.values()),
            max(len(f"{x:.4g}") for x in std_diff_b_train.values()),
            max(len(f"{x:.4g}") for x in std_diff_sgd_test.values()),
            max(len(f"{x:.4g}") for x in w_poly_rmse.values()),
            max(len(f"{x:.4g}") for x in w_sgd_rmse.values()),
            max(len(f"{x:.4g}") for x in y_poly_rmse.values()),
            max(len(f"{x:.4g}") for x in y_sgd_rmse.values()),
        )

        print(f"    {metrics[0][0]:<50} {metrics[0][1]:>{num_width}}  {metrics[0][2]:>{num_width}}")
        for row in metrics[1:]:
            print(f"    {row[0]:<50} {row[1]:>{num_width}.4g}  {row[2]:>{num_width}.4g}")
