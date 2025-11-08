import torch

class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient
    
    def clear(self):
        self._stored_gradient = None

def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad.T
        )
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.T.topk(num_candidates)
        if filter is not None:
            top_k_ids = top_k_ids.tolist()
            top_k_ids = [x for x in top_k_ids[0] if x not in filter]
    return top_k_ids



def get_embeddings(model):
    """Returns the wordpiece embedding module."""
    embeddings = model.get_input_embeddings()
    return embeddings

