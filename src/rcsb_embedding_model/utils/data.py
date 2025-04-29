from io import StringIO

import requests
import torch


def collate_seq_embeddings(batch_list):
    """
    Pads the tensors in a batch to the same size.

    Args:
        batch_list (list of torch.Tensor): A list of samples, where each sample is a tensor of shape (sequence_length, embedding_dim).

    Returns:
        tuple: A tuple containing:
            - padded_batch (torch.Tensor): A tensor of shape (batch_size, max_seq_length, embedding_dim), where each sample is padded to the max sequence length.
            - mask_batch (torch.Tensor): A tensor of shape (batch_size, max_seq_length) where padded positions are marked as False.
    """
    if batch_list[0] is None:
        return None
    device = batch_list[0].device  # Get the device of the input tensors
    max_len = max(sample.size(0) for sample in batch_list)  # Determine the maximum sequence length
    dim = batch_list[0].size(1)  # Determine the embedding dimension
    batch_size = len(batch_list)  # Determine the batch size

    # Initialize tensors for the padded batch and masks on the same device as the input tensors
    padded_batch = torch.zeros((batch_size, max_len, dim), dtype=batch_list[0].dtype, device=device)
    mask_batch = torch.ones((batch_size, max_len), dtype=torch.bool, device=device)

    for i, sample in enumerate(batch_list):
        seq_len = sample.size(0)  # Get the length of the current sequence
        padded_batch[i, :seq_len] = sample  # Pad the sequence with zeros
        mask_batch[i, :seq_len] = False  # Set mask positions for the actual data to False

    return padded_batch, mask_batch


def stringio_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return StringIO(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None


def concatenate_tensors(file_list, max_residues, dim=0):
    """
    Concatenates a list of tensors stored in individual files along a specified dimension.

    Args:
        file_list (list of str): List of file paths to tensor files.
        max_residues (int): Maximum number of residues allowed in the assembly
        dim (int): The dimension along which to concatenate the tensors. Default is 0.

    Returns:
        torch.Tensor: The concatenated tensor.
    """
    tensors = []
    total_residues = 0
    for file in file_list:
        try:
            tensor = torch.load(
                file,
                map_location=torch.device('cpu')
            )
            total_residues += tensor.shape[0]
            tensors.append(tensor)
        except Exception as e:
            continue
        if total_residues > max_residues:
            break
    if tensors and len(tensors) > 0:
        tensor_cat = torch.cat(tensors, dim=dim)
        return tensor_cat
    else:
        raise ValueError("No valid tensors were loaded to concatenate.")
