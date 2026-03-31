import sys
import os
sys.path.append(os.path.abspath(os.path.join("..", "")))

import torch
from dataset.diffusion_equations import DiffusionEquationDataset
from plots.loss_function import plot_losses
from plots.figures import plot_sample, plot_physical_property
from model import alpha_network, u_network
from loss_function.loss_function import Loss


def train_one_epoch(dataset, u_net, alpha_net, optimizer, loss_function):

    epoch_loss = 0

    for data_batch in dataset:

        x = data_batch["x"]
        xt = data_batch["xt"]
        u_xt = data_batch["u_xt"]

        # # predict alpha(x)
        # phys_coeff_pred = alpha_net(x=x)

        # # predict u(x,t)
        # phys_coeff_pred_repeated = phys_coeff_pred.repeat(1, data_batch["nt"]).flatten().unsqueeze(1)
        # xt_alpha = torch.cat([xt, phys_coeff_pred_repeated], dim=1)
        # u_pred = u_net(xt_alpha).reshape(data_batch["nx"], data_batch["nt"])

        # predict alpha(x)
        phys_coeff_pred = alpha_net(x=x)

        # predict u(x,t)
        u_pred = u_net(xt).reshape(data_batch["nx"], data_batch["nt"])


        optimizer.zero_grad()

        loss = loss_function(
            u_type=data_batch["u_type_txt"],
            x=x, 
            xt=xt,
            u_pred=u_pred,
            u_obs=u_xt,
            phys_coeff=data_batch["alpha"],
            phys_coeff_pred=phys_coeff_pred,
            nt=data_batch["nt"]
        )

        epoch_loss += loss

        loss.backward()
        
        optimizer.step()

    epoch_loss = epoch_loss / len(dataset)

    return epoch_loss


def train(dataset, u_net, alpha_net, loss_function, optimizer, epochs=3000):

    losses = []
    
    for epoch in range(epochs):

        epoch_loss = train_one_epoch(dataset=dataset, u_net=u_net, alpha_net=alpha_net, optimizer=optimizer, loss_function=loss_function)
        losses.append(epoch_loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss.item():.4e}")

    return losses


def main():

    # For PyTorch random
    torch.manual_seed(45)

    # Create Dataset
    dataset = DiffusionEquationDataset(n_samples=1, nx=100, nt=100,T=100, L=1)

    # Define PINNS
    alpha_net = alpha_network()
    u_net = u_network()

    # training
    u_net.train()
    alpha_net.train()

    lr = 1e-2
    optimizer = torch.optim.Adam(list(u_net.parameters()) + list(alpha_net.parameters()), lr=lr)
    loss = Loss(
        pde_coefficient=1.0,
        alpha_reg_coefficient=1.0,
        phys_property_coefficient=0.1
    )
    losses = train(
        loss_function=loss,
        optimizer=optimizer,
        dataset=dataset,
        u_net=u_net,
        alpha_net=alpha_net,
        epochs=3000
    )

    # Plot losses
    plot_losses(losses=[l.detach().numpy() for l in losses])

    # Plot sample
    u_net.eval()
    alpha_net.eval()
    sample = dataset[0]

    u_xt = u_net(xt=sample['xt'])
    sample_fig = plot_sample(sample, u_xt, xt_pred_mesh=sample)
    if not os.path.exists("diffusion_equation/figures"):
        os.makedirs("diffusion_equation/figures")
    sample_fig.savefig("diffusion_equation/figures/sample_prediction.png")

    # Plot physical property
    x = sample['x'].detach().numpy()[:,0]
    property_pred = alpha_net(x=sample['x']).detach().numpy()[:,0]
    property_true = sample["alpha"].detach().numpy()

    physical_property_fig = plot_physical_property(x=x, property_pred=property_pred, property_true=property_true)
    physical_property_fig.savefig("diffusion_equation/figures/physical_property_prediction.png")


if __name__ == "__main__":
    main()