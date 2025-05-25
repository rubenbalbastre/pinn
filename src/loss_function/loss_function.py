# loss_functions.py

import numpy as np
import torch
import torch.nn.functional as F

from utils import compute_waveform, get_batch, d_dt, d2_dt2, soln2orbit, one2two


def loss_function_case1_single_waveform(pred_sol, true_waveform, dt_data, model_params, NN_params=None,
                                         tsteps=None, loss_function="mae", coef_data=1.0, coef_weights=0.01, subset=250):
    mass_ratio = 0
    _, M, _, _ = model_params
    pred_waveform, _ = compute_waveform(dt_data, pred_sol, mass_ratio, M, model_params)

    if loss_function == "mae":
        loss = coef_data * F.l1_loss(pred_waveform[:subset], true_waveform[:subset]) + coef_weights * torch.sum(NN_params ** 2)
    elif loss_function == "mse":
        loss = F.mse_loss(pred_waveform, true_waveform)
    elif loss_function == "huber":
        loss = F.huber_loss(pred_waveform, true_waveform, delta=0.01)
    elif loss_function == "original":
        loss = torch.sum((true_waveform - pred_waveform) ** 2)
    else:
        raise ValueError("Unknown loss function")

    metric = F.l1_loss(pred_waveform[:subset], true_waveform[:subset])

    return {
        "loss": loss,
        "metric": metric,
        "pred_waveform": pred_waveform,
        "true_waveform": true_waveform,
        "tsteps": tsteps,
        "model_params": model_params
    }


def loss_function_case1(NN_params, processed_data, batch_size=None, loss_function_name="mae", subset=250):
    train_loss, train_metric = 0.0, 0.0
    test_loss, test_metric = 0.0, 0.0

    train_loss_information, test_loss_information = {}, {}
    train_subset = get_batch(processed_data["train"], batch_size)
    test_subset = get_batch(processed_data["test"], batch_size)

    for item in train_subset:
        pred_sol_train = item["nn_problem"].solve(NN_params)
        info = loss_function_case1_single_waveform(pred_sol_train, item["true_waveform"], item["dt_data"],
                                                   item["model_params"], NN_params,
                                                   tsteps=item["tsteps"], loss_function=loss_function_name, subset=subset)
        train_loss += abs(info["loss"])
        train_metric += abs(info["metric"])
        train_loss_information = info

    for item in reversed(test_subset):
        pred_sol_test = item["nn_problem"].solve(NN_params)
        info = loss_function_case1_single_waveform(pred_sol_test, item["true_waveform"], item["dt_data"],
                                                   item["model_params"], NN_params,
                                                   tsteps=item["tsteps"], loss_function=loss_function_name, subset=subset)
        test_loss += abs(info["loss"])
        test_metric += abs(info["metric"])
        test_loss_information = info

    N_train = len(train_subset)
    N_test = len(test_subset)

    return [
        train_loss / N_train,
        {
            "train_loss": train_loss / N_train,
            "test_loss": test_loss / N_test,
            "train_metric": train_metric / N_train,
            "test_metric": test_metric / N_test
        },
        train_loss_information,
        test_loss_information
    ]


def merge_info(a, b):
    a.update(b)
    return a


def loss_function_case2_single_waveform(pred_sol, waveform_real, dt_data, NN_params, model_params,
                                        reg_term=1e-1, stability_term=1.0, pos_ecc_term=1e1,
                                        dt2_term=1e2, dt_term=1e3, data_term=1.0, orbits_penalization=False,
                                        mass1_train=None, mass2_train=None, train_x_1=None, train_x_2=None):
    mass_ratio = model_params["q"]
    M = model_params["M"]
    pred_waveform_real, _ = compute_waveform(dt_data, pred_sol, mass_ratio, M, model_params)
    p = pred_sol[2, :]
    e = pred_sol[3, :]
    N = len(pred_waveform_real)

    orbit_loss = 0
    if orbits_penalization and len(pred_waveform_real) > int(0.95 * 1500):
        pred_orbit = soln2orbit(pred_sol, model_params)
        orbit_nn1, orbit_nn2 = one2two(pred_orbit, mass1_train, mass2_train)
        orbit_loss = F.mse_loss(
            torch.sqrt((train_x_1[:N] - orbit_nn1[0, :N])**2 + (train_x_2[:N] - orbit_nn1[1, :N])**2),
            torch.zeros(N)
        )

    loss = (data_term * F.l1_loss(pred_waveform_real, waveform_real[:N]) +
            dt_term * torch.sum(torch.clamp(d_dt(p, dt_data), min=0.0)**2) +
            dt2_term * torch.sum(torch.clamp(d2_dt2(p, dt_data), min=0.0)**2) +
            pos_ecc_term * torch.sum(torch.clamp(-e, min=0.0)**2) +
            pos_ecc_term * torch.sum(torch.clamp(-p, min=0.0)**2) +
            stability_term * torch.sum(torch.clamp(e[p >= 6 + 2 * e[0]] - e[0], min=0.0)**2) +
            orbit_loss +
            reg_term * torch.sum(NN_params**2))

    metric = torch.sqrt(torch.mean((pred_waveform_real - waveform_real[:N])**2))

    return {
        "loss": loss,
        "metric": metric,
        "pred_waveform": pred_waveform_real,
        "pred_solution": pred_sol
    }


def loss_function_case2(NN_params, tsteps_increment_bool, dataset_train, dataset_test):
    train_loss = train_metric = train_loss_complete = train_metric_complete = 0.0
    test_loss = test_metric = 0.0

    for wave_id, wave in dataset_train.items():
        pred_sol = wave["nn_problem"].solve(NN_params)
        train_results_i = loss_function_case2_single_waveform(pred_sol[:, tsteps_increment_bool],
                                                              wave["true_waveform"][tsteps_increment_bool],
                                                              wave["dt_data"], NN_params, wave["model_params"])

        train_results_i_complete = loss_function_case2_single_waveform(pred_sol,
                                                                       wave["true_waveform"],
                                                                       wave["dt_data"], NN_params, wave["model_params"])

        merge_info(train_results_i, wave)
        merge_info(train_results_i_complete, wave)

        train_loss += abs(train_results_i["loss"])
        train_loss_complete += abs(train_results_i_complete["loss"])
        train_metric += abs(train_results_i["metric"])
        train_metric_complete += abs(train_results_i_complete["metric"])

    for wave_id, wave in dataset_test.items():
        pred_sol = wave["nn_problem"].solve(NN_params)
        test_results_i_complete = loss_function_case2_single_waveform(pred_sol,
                                                                      wave["true_waveform"],
                                                                      wave["dt_data"], NN_params, wave["model_params"])

        merge_info(test_results_i_complete, wave)
        test_loss += abs(test_results_i_complete["loss"])
        test_metric += abs(test_results_i_complete["metric"])

    return [
        train_loss,
        {
            "train_loss": train_loss / len(dataset_train),
            "test_loss": test_loss / len(dataset_test),
            "train_loss_complete": train_loss_complete,
            "train_metric": train_metric / len(dataset_train),
            "train_metric_complete": train_metric_complete,
            "test_metric": test_metric / len(dataset_test),
        },
        train_results_i,
        train_results_i_complete,
        test_results_i_complete
    ]
