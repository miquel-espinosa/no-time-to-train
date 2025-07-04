import torch

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Reference: MoCo v2
    Gathers tensors from all GPUs and concatenates them.
    Works with single GPU or CPU training as well.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor
        
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output