import torch


def train_one_epoch(dataloader, encoder, decoder_dict, optimizer, loss_function):

    epoch_loss = 0.0

    for data_batch in dataloader:
        
        x = data_batch['x'].requires_grad_(True)  # (N, 1)
        xt = data_batch['xt'].requires_grad_(True)
        u_xt = data_batch['u_xt'].reshape(-1, 1).requires_grad_(True)  # (N, 1)
        u_type = data_batch['u_type'].reshape(-1,1).requires_grad_(True)
        u_type_txt = data_batch['u_type_txt']

        # Build input and encode
        u_type = xt.expand(u_xt.size(0), -1)
        inp = torch.cat([xt, u_xt, u_type], dim=1)  # (N, 4)
        z_mat = encoder(inp)  # (1, latent_dim)

        # Build decoder input & decode
        # z_repeated = z_mat.expand(xt.size(0), -1)  # [N, latent_dim]
        # xtz_mat = torch.cat([xt, z_repeated], dim=1)  # [N, latent_dim+1]
        # shared_decoding = decoder_dict['shared_decoder'](xtz_mat)

        # Predict physical coefficient (shared decoder + head)
        # property = decoder_dict['properties'](xtz_mat, u_type_txt)

        # Predict physical coefficient (shared decoder + head)
        z_repeated = z_mat.expand(x.size(0), -1)  # [N, latent_dim]
        xz_mat = torch.cat([x, z_repeated], dim=1)  # [N, latent_dim+1]
        property = decoder_dict['properties'](xz_mat, u_type_txt)

        # Predict physical measurement (shared decoder + head)
        property_repeated = property.repeat(1, data_batch["nt"]).flatten().unsqueeze(1)
        xt_alpha = torch.cat([xt, property_repeated], dim=1)
        u_pred = decoder_dict['measurements'](xt_alpha, u_type_txt)

        # Losses
        loss = loss_function(
            u_type=data_batch["u_type_txt"],
            x=x, 
            xt=xt,
            u_pred=u_pred,
            u_obs=u_xt,
            phys_coeff=data_batch["alpha"],
            phys_coeff_pred=property,
            nt=data_batch["nt"]
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def train(dataset, encoder, decoder, loss_function, optimizer, epochs=1000, batch_size=4):

    losses = []

    for epoch in range(epochs):
        epoch_loss = train_one_epoch(
            dataloader=dataset,
            encoder=encoder, 
            decoder_dict=decoder, 
            loss_function=loss_function, 
            optimizer=optimizer
        )
        losses.append(epoch_loss)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss:.4e}")
