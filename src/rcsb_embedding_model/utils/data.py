import os
import requests
import gzip
from io import StringIO, BytesIO

import torch
from requests import RequestException


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
        data = response.content
        if url.endswith('.bcif.gz'):
            with gzip.GzipFile(fileobj=BytesIO(data), mode='rb') as gz:
                decompressed_data = gz.read()
                return BytesIO(decompressed_data)
        if url.endswith('.gz'):
            compressed = BytesIO(data)
            with gzip.open(compressed, 'rt') as f:
                return StringIO(f.read())
        else:
            return StringIO(response.text)
    except RequestException as e:
        raise RuntimeError(f"Error fetching URL: {e}")
    except (OSError, gzip.BadGzipFile) as e:
        raise RuntimeError(f"Error decompressing gzip file: {e}")


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
        raise ValueError(f"No valid tensors were loaded to concatenate. {', '.join(file_list)}")

def adapt_csv_to_embedding_chain_stream(src_file, res_embedding_location):
    def __parse_row(row):
        r = row.split(",")
        return os.path.join(res_embedding_location, f"{r[0]}.{r[2]}.pt"), f"{r[0]}.{r[2]}"
    return tuple([__parse_row(r.strip()) for r in open(src_file) if len(r.split(",")) > 2])