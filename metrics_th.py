import torch

def compute_mse(predictions, targets):
    return torch.mean((predictions - targets) ** 2)

def compute_rmse(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))

def compute_mcrmse(all_score_predictions, all_score_targets):
    unique_classes = torch.unique(all_score_targets)
    num_classes = len(unique_classes)
    score_rmse = 0.
    
    for c in unique_classes:
        indices = (all_score_targets == c)
        score_predictions = all_score_predictions[indices]
        score_targets = all_score_targets[indices]
        score_rmse += 1 / num_classes * compute_rmse(score_predictions, score_targets)
    
    return score_rmse

def _accuracy_within_margin(score_predictions, score_target, margin, device):
    """ Returns the percentage of predicted scores that are within the provided margin from the target score. """
    return torch.sum(
        torch.where(
            torch.abs(score_predictions - score_target) <= margin,
            torch.ones(len(score_predictions), device=device),
            torch.zeros(len(score_predictions), device=device))).item() / len(score_predictions) * 100