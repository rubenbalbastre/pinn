import torch


def train_options_pinn_one_epoch(dataset, options_price_net, optimizer, loss_function):

    epoch_loss = 0

    for data_batch in dataset:

        St = data_batch["St"]
        V_St = data_batch["V_St"]

        # # predict V(S,t)
        V_pred = options_price_net(St).reshape(data_batch["nS"], data_batch["nt"])


        optimizer.zero_grad()

        loss = loss_function(
            St=St,
            V_pred=V_pred,
            V_obs=V_St
        )

        epoch_loss += loss

        loss.backward()
        
        optimizer.step()

    epoch_loss = epoch_loss / len(dataset)

    return epoch_loss


def train_options_pinn(dataset, options_price_net, loss_function, optimizer, epochs=3000):

    losses = []
    
    for epoch in range(epochs):

        epoch_loss = train_options_pinn_one_epoch(dataset=dataset, options_price_net=options_price_net, optimizer=optimizer, loss_function=loss_function)
        losses.append(epoch_loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss.item():.4e}")

    return losses

