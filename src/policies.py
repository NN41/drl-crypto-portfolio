# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNPolicy(nn.Module):
    def __init__(self, n_features: int, n_recent_periods: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=n_features, out_channels=2, kernel_size=(1, 3))
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=20, kernel_size=(1, n_recent_periods-2))
        self.conv3 = nn.Conv2d(in_channels=21, out_channels=1, kernel_size=1) # one extra in-channel for the previous portfolio vector
        self.cash_bias = nn.Parameter(torch.zeros(1)) # add a learnable cash score, initialized at 0
        self.n_recent_periods = n_recent_periods

    def forward(self, normalized_historical_prices: torch.Tensor, previous_portfolio_weights: torch.Tensor, availability_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            normalized_historical_prices: shape (B, F, M, N) or (F, M, N)
            previous_portfolio_weights: shape (B, M+1) or (M+1,), includes cash at index 0
            availability_mask: shape (B, M) or (M,), True where asset data exists
                B = batch size, F = n_features (=3), M = n_non_cash_assets, N = n_recent_periods (=50)
        """
        x = normalized_historical_prices
        w_prev = previous_portfolio_weights

        if x.dim() == 3: # add batch dimension if necessary
            x = x.unsqueeze(0)
        elif x.dim() != 4:
            raise ValueError(f"normalized_historical_prices must be 3d or 4d, got {x.dim()}")

        if w_prev.dim() == 1: # add batch dimension if necessary
            w_prev = w_prev.unsqueeze(0)
        elif w_prev.dim() != 2:
            raise ValueError(f"previous_portfolio_weights must be 1d or 2d, got {w_prev.dim()}")

        if availability_mask.dim() == 1:
            availability_mask = availability_mask.unsqueeze(0)

        B, _, M, N = x.shape
        assert N == self.n_recent_periods, f"Price history needs {self.n_recent_periods} periods, got {N}."
        assert availability_mask.shape == (B, M), f"Mask shape {availability_mask.shape} != expected ({B}, {M})"

        h1 = F.relu(self.conv1(x)) # (B, 2, M, N-2)
        h2 = F.relu(self.conv2(h1)) # (B, 20, M, 1)

        # inject previous non-cash portfolio weights as a feature map
        w_map = w_prev[:, 1:].unsqueeze(1).unsqueeze(-1) # (B, M) -> (B, 1, M, 1)
        h2 = torch.cat([w_map, h2], dim=1) # (B, 21, M, 1)

        scores = self.conv3(h2).squeeze(-1).squeeze(1) # (B, M) non-cash asset logits
        scores = scores.masked_fill(~availability_mask, float('-inf')) # unavailable assets get 0 weight after softmax

        cash_score = self.cash_bias.expand(B, 1) # broadcast (1,) to (B, 1) without copying
        logits = torch.cat([cash_score, scores], dim=1)
        weights = F.softmax(logits, dim=1)

        return weights
    
class EqualWeightPolicy:
    def __init__(self, n_non_cash_assets: int):
        self.n_non_cash_assets = n_non_cash_assets

    def __call__(self, price_history, previous_weights, availability_mask):
        # Equal weight across all available assets (0% cash)
        n_available = availability_mask.sum().item()
        weights = torch.zeros(self.n_non_cash_assets + 1, dtype=torch.float32)
        weights[1:] = availability_mask.float() / n_available
        return weights.unsqueeze(0)

## We don't need a policy for computing buy and hold performance metrics.
# class BuyAndHoldPolicy:
#     def __call__(self, normalized_historical_prices: torch.Tensor, previous_portfolio_weights: torch.Tensor):
#         x = normalized_historical_prices
#         w_prev = previous_portfolio_weights

#         if x.dim() == 3: # add batch dimension if necessary
#             x = x.unsqueeze(0)
#         elif x.dim() != 4:
#             raise ValueError(f"normalized_historical_prices must be 3d or 4d, got {x.dim()}")
        
#         if w_prev.dim() == 1: # add batch dimension if necessary
#             w_prev = w_prev.unsqueeze(0)
#         elif w_prev.dim() != 2:
#             raise ValueError(f"previous_portfolio_weights must be 1d or 2d, got {w_prev.dim()}")

#         price_relatives = 1 / x[:, -1, :, -2]
#         ones_tensor = torch.ones(price_relatives.shape[0], 1, device=x.device)
#         price_relatives = torch.concat([ones_tensor, price_relatives], dim=1)
#         w_prev_drifted = (price_relatives * w_prev) / torch.sum((price_relatives * w_prev), dim=1, keepdim=True)

#         return w_prev_drifted

# %%

if __name__ == '__main__':

    policy = CNNPolicy(n_features=3, n_recent_periods=50)

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nParameter Details:")
    for name, param in policy.named_parameters():
        print(f"  {name:30s} | Shape: {str(tuple(param.shape)):20s} | Params: {param.numel():,}")
        
    print("\nCritical Parameters:")
    print(f"  Cash bias value: {policy.cash_bias.item():.6f}")
    print(f"  Cash bias requires_grad: {policy.cash_bias.requires_grad}")

# %%
