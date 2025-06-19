import numpy as np
import matplotlib.pyplot as plt


# Example usage
def plot_sample(sample, u_xt_pred):
    
    u_xt = sample["u_xt"].detach().numpy()
    u_xt_pred = u_xt_pred.reshape(sample["nx"], sample["nt"]).detach().numpy()

    # Plot
    fig, ax = plt.subplots(figsize=(24, 4), nrows=1, ncols=3)

    im0 = ax[0].imshow(u_xt, aspect="auto", origin="lower", extent=[0, 1, 0, 0.1])
    fig.colorbar(im0, ax=ax[0], label="Temperature")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("Time")

    im1 = ax[1].imshow(u_xt_pred, aspect="auto", origin="lower", extent=[0, 1, 0, 0.1])
    fig.colorbar(im1, ax=ax[1], label="Temperature NN")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("Time")

    im2 = ax[2].imshow(np.abs(u_xt - u_xt_pred), aspect="auto", origin="lower", extent=[0, 1, 0, 0.1])
    fig.colorbar(im2, ax=ax[2], label="Temperature Absolute error")
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("Time")

    fig.suptitle(f"Heat diffusion (alpha: {sample['kind']})")

    plt.show()