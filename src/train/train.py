


def train_one_epoch(dataset, u_net, alpha_net, optimizer, loss_function):

    epoch_loss = 0

    for data_batch in dataset: #tqdm(dataset, desc=f"Batches"):

        # predict u(x,t), alpha(x)
        x = data_batch["x"].requires_grad_(True)
        xt = data_batch["xt"].requires_grad_(True)
        u_xt = data_batch["u_xt"].requires_grad_(True)

        u_pred = u_net(xt).reshape(data_batch["nx"], data_batch["nt"])

        phys_coeff_pred = alpha_net(x=x)

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


def train(dataset, u_net, alpha_net, loss_function, optimizer, epochs=1000):

    losses = []
    
    for epoch in range(epochs):

        epoch_loss = train_one_epoch(dataset=dataset, u_net=u_net, alpha_net=alpha_net, optimizer=optimizer, loss_function=loss_function)
        losses.append(epoch_loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss.item():.4e}")

