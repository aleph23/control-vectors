# code for computing conceptors borrowed from https://github.com/jorispos/ConceptorSteering/

import torch  # noqa: I001
import logging
from typing import Union, Optional, List
from hidden_state_data_manager import HiddenStateDataManager
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class ConceptorRepresentation:
    is_low_rank: bool
    full: Optional[torch.Tensor] = None
    U: Optional[torch.Tensor] = None
    s: Optional[torch.Tensor] = None

def scale_conceptor(C: torch.Tensor, aperture_scaling_factor: float) -> torch.Tensor:
    """
    Scales a conceptor matrix using the given aperture scaling factor.

    Args:
        C: Conceptor matrix of shape (d, d)
        aperture_scaling_factor: Positive scaling factor

    Returns:
        Scaled conceptor matrix of shape (d, d)
    """
    return C @ torch.inverse(C + (aperture_scaling_factor**-2) * (torch.eye(C.shape[0], device=C.device) - C))


def rescale_conceptor(C, prev_alpha, new_alpha):
    return scale_conceptor(C, new_alpha / prev_alpha)


def compute_conceptor(X, aperture):
    """
    Computes the conceptor matrix for a given input matrix X.
    (PyTorch version)

    Parameters:
    - X (torch.Tensor): Input matrix of shape (n_samples, n_features).
    - torch.Tensor: Conceptor matrix of shape (n_features, n_features).
    """
    R = torch.matmul(X.T, X) / X.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(R)
    C = eigenvectors * (eigenvalues / (eigenvalues + (aperture ** (-2)) * torch.ones(eigenvalues.shape, device=X.device))) @ eigenvectors.T
    return C

def combine_conceptors_and(C1: torch.Tensor, C2: torch.Tensor) -> torch.Tensor:
    """
    Combines two conceptors C1 and C2 using the given formula. (AND operation, does not work so well)

    Parameters:
    - C1 (torch.Tensor): First conceptor tensor of shape (n_features, n_features).
    - C2 (torch.Tensor): Second conceptor tensor of shape (n_features, n_features).

    Returns:
    - torch.Tensor: Combined conceptor tensor of shape (n_features, n_features).
    """
    I = torch.eye(C1.shape[0], device=C1.device)  # Identity matrix
    C1_inv = torch.inverse(C1)
    C2_inv = torch.inverse(C2)
    combined_inv = C1_inv + C2_inv - I
    combined = torch.inverse(combined_inv)
    return combined


def combine_conceptors(C1, C2):
    """
    TODO: what is this?
    Combines two conceptors C1 and C2 using the given new formula. (OR operation which works much better than AND)

    Parameters:
    - C1 (torch.Tensor): First conceptor tensor of shape (n_features, n_features).
    - C2 (torch.Tensor): Second conceptor tensor of shape (n_features, n_features).

    Returns:
    - torch.Tensor: Combined conceptor tensor of shape (n_features, n_features).
    """
    I = torch.eye(C1.shape[0], device=C1.device)  # Identity matrix
    I_C1_inv = torch.inverse(I - C1)
    I_C2_inv = torch.inverse(I - C2)
    combined_inv = I_C1_inv + I_C2_inv - I
    combined = torch.inverse(combined_inv)
    result = I - combined
    return result


def compute_conceptor_low_rank(X: torch.Tensor, aperture: float, rank: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes a low-rank approximation of the conceptor matrix for given input matrix X.

    Parameters:
    - X (torch.Tensor): Input matrix of shape (n_samples, n_features).
    - aperture (float): Aperture parameter.
    - rank (int): Rank for the approximation.

    Returns:
    - U (torch.Tensor): Left singular vectors of shape (d, rank).
    - S_c (torch.Tensor): Scaled singular values of shape (rank,).
    """
    # Compute covariance matrix
    R = torch.matmul(X.T, X) / X.shape[0]

    # Perform truncated SVD
    U, S, V = torch.svd_lowrank(R, q=rank)

    # Scale singular values as per conceptor formula
    S_c = S / (S + aperture ** (-2))

    return U, S_c


def compute_low_rank_conceptor_automatic(X: torch.Tensor, aperture: float, variance_thres: float=0.99) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes a low-rank approximation of the conceptor matrix using a variance threshold
    to select the rank.

    Parameters:
        X (torch.Tensor): Input matrix of shape (n_samples, n_features).
        aperture (float): Aperture parameter.
        variance_thres (float): Desired variance retention.

    Returns:
        U_k (torch.Tensor): Left singular vectors (n_features, k).
        s_k (torch.Tensor): Singular values (k,).
    """
    # Compute the correlation matrix
    R = torch.matmul(X.T, X) / X.shape[0]

    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(R)

    # Reverse to get descending order
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    # Compute scaling factors
    scaling_factors = eigenvalues / (eigenvalues + aperture ** (-2))

    # Cumulative variance
    cumulative_variance = torch.cumsum(scaling_factors, dim=0)
    total_variance = cumulative_variance[-1]
    variance_ratios = cumulative_variance / total_variance

    # Determine k
    k = torch.searchsorted(variance_ratios, variance_thres).item() + 1

    U_k = eigenvectors[:, :k]
    s_k = scaling_factors[:k]

    return U_k, s_k


def compute_optimal_low_rank_conceptor(C: torch.Tensor, thres: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes a low-rank approximation of a conceptor matrix by keeping only
    significant singular values.

    Args:
        C: Original conceptor matrix
        thres: Relative thres for keeping singular values
                  (compared to largest singular value)

    Returns:
        U: Left singular vectors (d × k)
        s: Singular values (k,)
        where k is the determined rank
    """
    U, s, _ = torch.svd(C)

    # Find optimal rank by looking at singular value decay
    max_sv = s[0]
    mask = s >= (thres * max_sv)
    k = mask.sum().item()

    # Keep only the top k components
    U_k = U[:, :k]
    s_k = s[:k]

    return U_k, s_k


def reconstruct_conceptor(U: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Reconstructs full conceptor matrix from low-rank representation."""
    return U @ torch.diag(s) @ U.T


class ConceptorAnalyzer:
    """
    Computes conceptors for each dataset (class) and layer, optionally after mean-centering.

    Produces conceptors[class_idx][layer_idx] => (d x d) or None,
    plus a parallel structure means[class_idx][layer_idx] => (d,) or None
    if we want to store the mean used.

    The 'center' can be:
      - 'none': no mean-centering
      - 'local': subtract each dataset's own mean
      - 'baseline': subtract the baseline dataset's mean from all other classes
         (assuming class_idx=0 is 'baseline').
    """

    def __init__(
            self,
            hidden_state_data_manager: "HiddenStateDataManager",
            skip_early_layers: int,
            skip_late_layers: int,
            aperture: float,
            center: str,
            lora: bool,
            lora_method: Optional[str],
            rank: Optional[int],
            variance_thres: Optional[float],
            thres: Optional[float]
    ):
        """
        Args:
        hidden_state_data_manager: The data manager that supplies get_datasets(layer).
        skip_early_layers (int): # of initial layers to skip (may be fraction).
        skip_late_layers (int): # of final layers to skip (may be fraction).
        aperture (float): Aperture parameter for the conceptor formula (must be > 0).
        center (str): "none", "local", "baseline" -- how we perform mean-centering.
        """
        if aperture <= 0:
            raise ValueError("Aperture must be positive")
        self.hidden_state_data_manager = hidden_state_data_manager

        self.num_layers = hidden_state_data_manager.get_num_layers()
        self.num_dataset_types = hidden_state_data_manager.get_num_dataset_types()

        if isinstance(skip_early_layers, float) and 0 < skip_early_layers < 1:
            skip_early_layers = round(skip_early_layers * self.num_layers)
        if isinstance(skip_late_layers, float) and 0 < skip_late_layers < 1:
            skip_late_layers = round(skip_late_layers * self.num_layers)

        if skip_early_layers + skip_late_layers >= self.num_layers:
            raise ValueError("Skipping all layers is fast, but you'll find the results unsatisfying (start + end >= total layers).")

        self.skip_early_layers = skip_early_layers
        self.skip_late_layers = skip_late_layers

        self.aperture = aperture
        self.center = center.lower()
        if self.center not in ["none", "local", "baseline"]:
            raise ValueError(f"Invalid center: {self.center}")

        self.lora = lora
        self.lora_method = None
        self.rank = None
        self.variance_thres = variance_thres
        self.thres = thres

        if self.lora:
            if lora_method not in ["manual", "automatic", "optimal"]:
                raise ValueError("lora_method must be one of 'manual', 'automatic', or 'optimal' when lora is True")
            self.lora_method = lora_method
            if lora_method == "manual":
                if rank is None:
                    raise ValueError("When lora_method is 'manual', 'rank' must be specified")
                self.rank = rank
            elif lora_method == "automatic":
                if variance_thres is None:
                    raise ValueError("When lora_method is 'automatic', 'variance_thres' must be specified")
                self.variance_thres = variance_thres
            else:
                if thres is None:
                    raise ValueError("When lora_method is 'optimal', 'thres' must be specified")
                self.thres = thres

        self.conceptors: List[List[Optional[ConceptorRepresentation]]] = []
        self.means: List[List[Optional[torch.Tensor]]] = []
        self._compute_conceptors_all()

    def _compute_baseline_means(self) -> List[Optional[torch.Tensor]]:
        baseline_means_by_layer: List[Optional[torch.Tensor]] = [None] * self.num_layers

        if self.center == "baseline" and self.num_dataset_types > 0:
            for layer_idx in range(self.skip_early_layers, self.num_layers - self.skip_late_layers):
                X_base = self.hidden_state_data_manager.get_datasets(layer_idx)[0]
                X_base = X_base.float().cpu()
                baseline_means_by_layer[layer_idx] = X_base.mean(dim=0, keepdim=True)
        return baseline_means_by_layer

    def _center_X(
        self,
        X: torch.Tensor,
        class_idx: int,
        layer_idx: int,
        baseline_means_by_layer: List[Optional[torch.Tensor]],
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        if self.center == "none":
            return X, None
        if self.center == "local":
            mean_vec = X.mean(dim=0, keepdim=True)
            return X - mean_vec, mean_vec
        if self.center == "baseline":
            if class_idx == 0:
                return X, None
            mean_vec = baseline_means_by_layer[layer_idx]
            return X - mean_vec, mean_vec
        raise ValueError(f"Unknown center {self.center}")

    def _compute_conceptor_for_X(self, X: torch.Tensor) -> ConceptorRepresentation:
        if self.lora:
            if self.lora_method == "manual":
                if self.rank is None or not 1 <= self.rank <= X.shape[1]:
                    raise ValueError("rank must be between 1 and the feature dimension")
                U_k, s_k = compute_conceptor_low_rank(X, self.aperture, rank=self.rank)
            elif self.lora_method == "automatic":
                U_k, s_k = compute_low_rank_conceptor_automatic(X, self.aperture, variance_thres=self.variance_thres)
            else:
                C = compute_conceptor(X, self.aperture)
                U_k, s_k = compute_optimal_low_rank_conceptor(C, thres=self.thres)
            return ConceptorRepresentation(is_low_rank=True, U=U_k.cpu(), s=s_k.cpu())
        else:
            C = compute_conceptor(X, self.aperture)
            return ConceptorRepresentation(is_low_rank=False, full=C.cpu())

    def _compute_conceptors_all(self):
        self.conceptors = [[None for _ in range(self.num_layers)]
                           for _ in range(self.num_dataset_types)]
        self.means = [[None for _ in range(self.num_layers)]
                      for _ in range(self.num_dataset_types)]

        baseline_means_by_layer = self._compute_baseline_means()

        total_computations = (self.num_layers - self.skip_early_layers - self.skip_late_layers) * self.num_dataset_types
        with tqdm(total=total_computations, desc="Computing conceptors") as pbar:
            for class_idx in range(self.num_dataset_types):
                for layer_idx in range(self.skip_early_layers, self.num_layers - self.skip_late_layers):
                    X = self.hidden_state_data_manager.get_datasets(layer_idx)[class_idx]
                    X = X.float().cpu()

                    X_centered, mean_vec = self._center_X(X, class_idx, layer_idx, baseline_means_by_layer)

                    try:
                        conceptor_repr = self._compute_conceptor_for_X(X_centered)
                        self.conceptors[class_idx][layer_idx] = conceptor_repr
                    except RuntimeError as e:
                        logging.error(f"Error computing conceptor at layer={layer_idx} class={class_idx}: {e}")

                    if mean_vec is not None:
                        self.means[class_idx][layer_idx] = mean_vec.squeeze(0).cpu()

                    pbar.update(1)

    def get_conceptor(self, class_idx: int, layer_idx: int) -> Optional[torch.Tensor]:
        """Returns the conceptor for the given class/layer, or None if not computed."""
        return self.conceptors[class_idx][layer_idx]

    def get_mean(self, class_idx: int, layer_idx: int) -> Optional[torch.Tensor]:
        """Returns the (d,)-shaped mean vector used for that (class,layer), or None if none was used."""
        return self.means[class_idx][layer_idx]
