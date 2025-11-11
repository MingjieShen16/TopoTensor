import numpy as np

# times: increasing sample times t0...tN
# deltaO: shape (N+1, d) inferred order flows at those times

def piecewise_constant_values(times, deltaO, t_grid):
    """Return RCLL piecewise-constant path evaluated on t_grid."""
    # assume t_grid sorted and within [times[0], times[-1]]
    idx = np.searchsorted(times, t_grid, side='right') - 1
    idx = np.clip(idx, 0, len(times)-1)
    return deltaO[idx]   # shape (len(t_grid), d)

# sliding windows
def extract_windows_from_grid(values, window_size, stride):
    # values shape (Tsteps, d)
    n_windows = (len(values) - window_size) // stride + 1
    windows = np.zeros((n_windows, window_size, values.shape[1]))
    for i in range(n_windows):
        windows[i] = values[i*stride : i*stride + window_size]
    return windows

# Example usage:
# choose a regular evaluation grid (e.g., every 100ms)
t_grid = np.linspace(times[0], times[-1], num=10000)
vals = piecewise_constant_values(times, deltaO, t_grid)
windows = extract_windows_from_grid(vals, window_size=200, stride=50)


# build jump-size channel aligned with t_grid
jump_sizes = np.zeros((len(t_grid), d))
for k in range(1, len(times)):
    # find grid index equal to times[k]
    gi = np.searchsorted(t_grid, times[k])
    jump_sizes[gi] = deltaO[k] - deltaO[k-1]
# concatenate as extra features
vals_with_jumps = np.concatenate([vals, jump_sizes], axis=1)
