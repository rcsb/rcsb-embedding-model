
def is_distributed(devices, num_nodes) -> bool:
    """Multi-process (DDP) run → the post-inference barrier needs more than NCCL's 30-min default."""
    if num_nodes > 1:
        return True
    if isinstance(devices, int):
        return devices != 1            # >1, or -1 (= all visible)
    if isinstance(devices, (list, tuple)):
        return len(devices) > 1
    return True