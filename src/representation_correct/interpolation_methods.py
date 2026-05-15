import torch
from typing import Dict, Union, List, Any


def solve_pareto_w(matrices_dict, c_matrices_dict, preferences, beta=0.1):
    numerator = 0
    denominator = 0

    for i, (task_name, W_t) in enumerate(matrices_dict.items()):
        p_t = preferences[i]
        C_t = c_matrices_dict[task_name]
        reg_C = C_t + beta * torch.eye(C_t.shape[0], device=C_t.device)

        numerator += p_t * (W_t @ reg_C)
        denominator += p_t * reg_C
    W_p = torch.linalg.solve(denominator.t(), numerator.t()).t()
    return W_p


@torch.no_grad()
def linear_pareto_interpolation(
        matrix_dict: Dict[str, Any],
        weights: torch.Tensor,
        Cmm_matrices: Dict[str, Any],
        orth_reg: float = 0.1
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Wrapper for ReACT Pareto aggregation compatible with fusion_bench.
    Handles both flat (global) and nested (layer-wise) structures.
    """
    if not matrix_dict:
        return {}
    # Check structure of the first element in matrix_dict
    first_val = next(iter(matrix_dict.values()))

    # Case 1: Nested Structure (Layer-wise W)
    if isinstance(first_val, dict):
        interpolated = {}
        layer_names = first_val.keys()

        # Check structure of Cmm_matrices to see if it matches W or is global
        first_c_val = next(iter(Cmm_matrices.values()))
        is_c_nested = isinstance(first_c_val, dict)

        for layer in layer_names:
            # Slice W for this layer: {task: W_layer}
            layer_Ws = {k: v[layer] for k, v in matrix_dict.items()}

            # Slice C for this layer
            if is_c_nested:
                # If C is also layer-wise, extract the specific layer
                layer_Cs = {k: v[layer] for k, v in Cmm_matrices.items()}
            else:
                # If C is global (flat Tensor), reuse it for all layers
                # Note: This broadcasts the global covariance to all layers.
                layer_Cs = Cmm_matrices

            # Recursive call
            interpolated[layer] = linear_pareto_interpolation(
                layer_Ws, weights, layer_Cs, orth_reg
            )
        return interpolated

    # Case 2: Flat Structure (Global W)
    # Ensure weights are on the correct device
    device = first_val.device
    preferences = weights.to(device)

    # Call the core mathematical implementation
    return solve_pareto_w(
        matrices_dict=matrix_dict,
        c_matrices_dict=Cmm_matrices,
        preferences=preferences,
        beta=orth_reg
    )
