import matplotlib.pyplot as plt


def plot_losses(losses):

    plt.plot(losses)
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
