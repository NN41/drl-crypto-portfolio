# %%
import matplotlib.pyplot as plt
from src.train_utils import geometrically_sample_batch_start_indices

result = geometrically_sample_batch_start_indices(
    n_samples=100000,
    n_available_periods=32504,
    batch_size=50,
    geometric_parameter=5e-5,
    n_recent_periods=50
)

result2 = geometrically_sample_batch_start_indices(
    n_samples=100000,
    n_available_periods=32504,
    batch_size=50,
    geometric_parameter=1e-6,
    n_recent_periods=50
)

result3 = geometrically_sample_batch_start_indices(
    n_samples=100000,
    n_available_periods=56000,
    batch_size=50,
    geometric_parameter=5e-5,
    n_recent_periods=50
)

plt.hist(result, alpha=0.6, density=True, bins=20)
plt.hist(result2, alpha=0.6, density=True, bins=20)
plt.hist(result3, alpha=0.6, density=True, bins=40)


# %%




plt.hist(result)
# %%
