import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from src.physics.orbital_mechanics import compute_waveform, get_batch, d_dt, d2_dt2, soln2orbit, one2two


def loss_function_case1_single_waveform(
    pred_sol: torch.Tensor,
    true_waveform: torch.Tensor,
    dt_data: float,
    model_params: Tuple[float, float, float, float],
    NN_params: Optional[torch.Tensor] = None,
    tsteps: Optional[torch.Tensor] = None,
    loss_function: str = "mae",
    coef_data: float = 1.0,
    coef_weights: float = 0.01,
    subset: int = 250,
) -> Dict[str, Any]:
    mass_ratio = 0
    _, M, _, _ = model_params
    pred_waveform, _ = compute_waveform(dt_data, pred_sol, mass_ratio, M, model_params)

    # Select loss function
    if loss_function == "mae":
        loss = coef_data * F.l1_loss(pred_waveform[:subset], true_waveform[:subset])
        if NN_params is not None:
            loss += coef_weights * torch.sum(NN_params ** 2)
    elif loss_function == "mse":
        loss = F.mse_loss(pred_waveform[:subset], true_waveform[:subset])
    elif loss_function == "huber":
        loss = F.smooth_l1_loss(pred_waveform[:subset], true_waveform[:subset], beta=0.01)  # huber loss alias
    elif loss_function == "original":
        loss = torch.sum((true_waveform[:subset] - pred_waveform[:subset]) ** 2)
    else:
        raise ValueError(f"Unknown loss function '{loss_function}'")

    metric = F.l1_loss(pred_waveform[:subset], true_waveform[:subset])

    return {
        "loss": loss,
        "metric": metric,
        "pred_waveform": pred_waveform,
        "true_waveform": true_waveform,
        "tsteps": tsteps,
        "model_params": model_params,
    }


def loss_function_case1(
    NN_params: torch.Tensor,
    processed_data: Dict[str, Any],
    batch_size: Optional[int] = None,
    loss_function_name: str = "mae",
    subset: int = 250,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, Any], Dict[str, Any]]:
    train_loss, train_metric = 0.0, 0.0
    test_loss, test_metric = 0.0, 0.0

    train_loss_information, test_loss_information = {}, {}
    train_subset = get_batch(processed_data["train"], batch_size)
    test_subset = get_batch(processed_data["test"], batch_size)

    for item in train_subset:
        pred_sol_train = item["nn_problem"].solve(NN_params)
        info = loss_function_case1_single_waveform(
            pred_sol_train,
            item["true_waveform"],
            item["dt_data"],
            item["model_params"],
            NN_params,
            tsteps=item.get("tsteps"),
            loss_function=loss_function_name,
            subset=subset,
        )
        train_loss += info["loss"].item()
        train_metric += info["metric"].item()
        train_loss_information = info

    for item in reversed(test_subset):
        pred_sol_test = item["nn_problem"].solve(NN_params)
        info = loss_function_case1_single_waveform(
            pred_sol_test,
            item["true_waveform"],
            item["dt_data"],
            item["model_params"],
            NN_params,
            tsteps=item.get("tsteps"),
            loss_function=loss_function_name,
            subset=subset,
        )
        test_loss += info["loss"].item()
        test_metric += info["metric"].item()
        test_loss_information = info

    N_train = len(train_subset)
    N_test = len(test_subset)

    return (
        torch.tensor(train_loss / N_train),
        {
            "train_loss": train_loss / N_train,
            "test_loss": test_loss / N_test,
            "train_metric": train_metric / N_train,
            "test_metric": test_metric / N_test,
        },
        train_loss_information,
        test_loss_information,
    )


def merge_info(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    a.update(b)
    return a


def loss_function_case2_single_waveform(
    pred_sol: torch.Tensor,
    waveform_real: torch.Tensor,
    dt_data: float,
    NN_params: torch.Tensor,
    model_params: Dict[str, Any],
    reg_term: float = 1e-1,
    stability_term: float = 1.0,
    pos_ecc_term: float = 1e1,
    dt2_term: float = 1e2,
    dt_term: float = 1e3,
    data_term: float = 1.0,
    orbits_penalization: bool = False,
    mass1_train: Optional[float] = None,
    mass2_train: Optional[float] = None,
    train_x_1: Optional[torch.Tensor] = None,
    train_x_2: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    mass_ratio = model_params["q"]
    M = model_params["M"]
    pred_waveform_real, _ = compute_waveform(dt_data, pred_sol, mass_ratio, M, model_params)
    p = pred_sol[2, :]
    e = pred_sol[3, :]
    N = len(pred_waveform_real)

    orbit_loss = torch.tensor(0.0, device=pred_sol.device)
    if orbits_penalization and N > int(0.95 * 1500) and train_x_1 is not None and train_x_2 is not None:
        pred_orbit = soln2orbit(pred_sol, model_params)
        orbit_nn1, orbit_nn2 = one2two(pred_orbit, mass1_train, mass2_train)
        orbit_loss = F.mse_loss(
            torch.sqrt((train_x_1[:N] - orbit_nn1[0, :N]) ** 2 + (train_x_2[:N] - orbit_nn1[1, :N]) ** 2),
            torch.zeros(N, device=pred_sol.device),
        )

    loss = (
        data_term * F.l1_loss(pred_waveform_real, waveform_real[:N])
        + dt_term * torch.sum(torch.clamp(d_dt(p, dt_data), min=0.0) ** 2)
        + dt2_term * torch.sum(torch.clamp(d2_dt2(p, dt_data), min=0.0) ** 2)
        + pos_ecc_term * torch.sum(torch.clamp(-e, min=0.0) ** 2)
        + pos_ecc_term * torch.sum(torch.clamp(-p, min=0.0) ** 2)
        + stability_term * torch.sum(torch.clamp(e[p >= 6 + 2 * e[0]] - e[0], min=0.0) ** 2)
        + orbit_loss
        + reg_term * torch.sum(NN_params**2)
    )

    metric = torch.sqrt(torch.mean((pred_waveform_real - waveform_real[:N]) ** 2))

    return {
        "loss": loss,
        "metric": metric,
        "pred_waveform": pred_waveform_real,
        "pred_solution": pred_sol,
    }


def loss_function_case2(
    NN_params: torch.Tensor,
    tsteps_increment_bool: torch.Tensor,
    dataset_train: Dict[Any, Any],
    dataset_test: Dict[Any, Any],
):
    train_loss = 0.0
    train_metric = 0.0
    train_loss_complete = 0.0
    train_metric_complete = 0.0
    test_loss = 0.0
    test_metric = 0.0

    for wave_id, wave in dataset_train.items():
        pred_sol = wave["nn_problem"].solve(NN_params)
        train_results_i = loss_function_case2_single_waveform(
            pred_sol[:, tsteps_increment_bool],
            wave["true_waveform"][tsteps_increment_bool],
            wave["dt_data"],
            NN_params,
            wave["model_params"],
        )

        train_results_i_complete = loss_function_case2_single_waveform(
            pred_sol,
            wave["true_waveform"],
            wave["dt_data"],
            NN_params,
            wave["model_params"],
        )

        merge_info(train_results_i, wave)
        merge_info(train_results_i_complete, wave)

        train_loss += train_results_i["loss"].item()
        train_loss_complete += train_results_i_complete["loss"].item()
        train_metric += train_results_i["metric"].item()
        train_metric_complete += train_results_i_complete["metric"].item()

    for wave_id, wave in dataset_test.items():
        pred_sol = wave["nn_problem"].solve(NN_params)
        test_results_i_complete = loss_function_case2_single_waveform(
            pred_sol,
            wave["true_waveform"],
            wave["dt_data"],
            NN_params,
            wave["model_params"],
        )

        merge_info(test_results_i_complete, wave)
        test_loss += test_results_i_complete["loss"].item()
        test_metric += test_results_i_complete["metric"].item()

    n_train = len(dataset_train)
    n_test = len(dataset_test)

    return [
        torch.tensor(train_loss),
        {
            "train_loss": train_loss / n_train,
            "test_loss": test_loss / n_test,
            "train_loss_complete": train_loss_complete,
            "train_metric": train_metric / n_train,
            "train_metric
        }
        ]