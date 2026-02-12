import numpy as np
import matplotlib.pyplot as plt


# Example usage
def plot_sample(sample, u_xt_pred, xt_pred_mesh):
    
    x = sample['x'].detach().numpy()
    t = sample['t'].detach().numpy()
    u_xt = sample["u_xt"].detach().numpy()

    x_pred = xt_pred_mesh['x'].detach().numpy()
    t_pred = xt_pred_mesh['t'].detach().numpy()
    u_xt_pred = u_xt_pred.reshape(sample["nx"], sample["nt"]).detach().numpy()

    # Plot
    fig, ax = plt.subplots(figsize=(24, 4), nrows=1, ncols=3)

    im0 = ax[0].imshow(u_xt, aspect="auto", origin="lower", extent=[x.min(), x.max(), t.min(), t.max()])
    fig.colorbar(im0, ax=ax[0], label="True Measurement")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("Time")

    im1 = ax[1].imshow(u_xt_pred, aspect="auto", origin="lower", extent=[x_pred.min(), x_pred.max(), t_pred.min(), t_pred.max()])
    fig.colorbar(im1, ax=ax[1], label="NN Prediction")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("Time")

    im2 = ax[2].imshow(np.abs(u_xt - u_xt_pred), aspect="auto", origin="lower", extent=[x_pred.min(), x_pred.max(), t_pred.min(), t_pred.max()])
    fig.colorbar(im2, ax=ax[2], label="Measurement Absolute error")
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("Time")

    fig.suptitle(f"{sample["u_type_txt"].capitalize()} (physical coefficient: {sample['kind']})")

    plt.show()
    

def plot_physical_property(x, property_true, property_pred):

    plt.plot(x, property_true, label="true", color='b')
    plt.xlabel("x")
    plt.ylabel("property")
    plt.plot(x, property_pred, label='pred', color='r', linestyle='--')
    plt.legend()
    plt.title("Physical Property Prediction")
    plt.grid()

    plt.show()