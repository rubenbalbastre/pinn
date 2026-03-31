import torch


def train_one_epoch_bcd(dataset, u_net, alpha_net, optimizer_u, optimizer_alpha, loss_function_alpha, loss_function_u):

    epoch_loss = 0

    for data_batch in dataset:

        x = data_batch["x"]
        xt = data_batch["xt"]
        u_xt = data_batch["u_xt"]
        nt = data_batch["nt"]

        ### === PHASE 1: Update u_net, freeze alpha_net === ###
        for param in alpha_net.parameters():
            param.requires_grad = False
        for param in u_net.parameters():
            param.requires_grad = True

        optimizer_u.zero_grad()

        phys_coeff_pred = alpha_net(x=x)#.detach().clone().requires_grad_()   # freeze alpha_net

        u_pred = u_net(xt).reshape(data_batch["nx"], nt)

        loss_u = loss_function_u(
            u_type=data_batch["u_type_txt"],
            x=x, 
            xt=xt,
            u_pred=u_pred,
            u_obs=u_xt,
            phys_coeff=data_batch["alpha"],
            phys_coeff_pred=phys_coeff_pred,
            nt=nt
        )

        loss_u.backward()
        optimizer_u.step()

        ### === PHASE 2: Update alpha_net, freeze u_net === ###
        for param in alpha_net.parameters():
            param.requires_grad = True
        for param in u_net.parameters():
            param.requires_grad = False

        optimizer_alpha.zero_grad()

        u_pred = u_net(xt).reshape(data_batch["nx"], nt)#.detach().clone().requires_grad_() # freeze u_net

        phys_coeff_pred = alpha_net(x=x)

        loss_alpha = loss_function_alpha(
            u_type=data_batch["u_type_txt"],
            x=x, 
            xt=xt,
            u_pred=u_pred, 
            u_obs=u_xt,
            phys_coeff=data_batch["alpha"],
            phys_coeff_pred=phys_coeff_pred,
            nt=nt
        )

        loss_alpha.backward()
        optimizer_alpha.step()

        epoch_loss += (loss_u + loss_alpha)

    epoch_loss /= len(dataset)
    return epoch_loss


def train_bcd(dataset, u_net, alpha_net, loss_function_u, loss_function_alpha, optimizer_u, optimizer_alpha, epochs=3000):

    losses = []
    
    for epoch in range(epochs):

        epoch_loss = train_one_epoch_bcd(dataset=dataset, u_net=u_net, alpha_net=alpha_net, optimizer_u=optimizer_u, optimizer_alpha=optimizer_alpha, loss_function_u=loss_function_u, loss_function_alpha=loss_function_alpha)
        losses.append(epoch_loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss.item():.4e}")

    return losses