import torch


def train_one_epoch(dataloader, encoder, decoder_dict, optimizer, loss_function):

    epoch_loss = 0.0

    for data_batch in dataloader:
        
        x = data_batch['x'].requires_grad_(True)  # (N, 1)
        t = data_batch['t'].requires_grad_(True)  # (N, 1)
        u_xt = data_batch['u_xt'].reshape(-1, 1)  # (N, 1)
        u_type = data_batch['u_type']  # (N,) or one-hot (N, K)

        # Build input and encode
        xt = torch.cat([x, t], dim=1)
        inp = torch.cat([xt, u_xt, u_type], dim=1)  # (N, D)
        z = encoder(inp)  # (N, latent_dim)

        # Predict physical coefficient (shared decoder + head)
        phys_coeff_pred = decoder_dict['coefficient'](z)
        u_pred = decoder_dict['u'](z)

        # Losses
        loss = loss_function(
            data_type=data_batch["data_type"],
            x=x, 
            xt=xt,
            u_pred=u_pred,
            u_obs=u_xt,
            phys_coeff_pred=phys_coeff_pred.requires_grad_(True),
            nt=data_batch["nt"]
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def train_model(dataset, encoder, decoder, loss_function, optimizer, epochs=1000, batch_size=4):

    losses = []

    for epoch in range(epochs):
        epoch_loss = train_one_epoch(encoder, decoder, dataset, loss_function, optimizer)
        losses.append(epoch_loss)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss:.4e}")
