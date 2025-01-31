import torch

# Compute the loss for a single batch
def calc_loss_batch(input_batch, target_batch, model, device):
    # The transfer to a given device allows us to transfer the data to a GPU
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0,1), target_batch.flatten()
    )
    return loss


# Compute the loss over all the batches sampled by a given data loader
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        # Iterates over all batches if no fixed num_batches is given
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches 
        # in the data loader if num_batches exceeds the number of batches 
        # in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        # Only proceed a maximum of num_batches
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    
    # Average the loss over all batches
    return total_loss/num_batches