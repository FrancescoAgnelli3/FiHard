
def normalize(data, stats, normalize_type="normalize_feature"):
    if 'normalize' in normalize_type:
        data = 2 * (data - stats["min"]) / (stats["max"] - stats["min"]) - 1
    elif 'meanstd' in normalize_type:
        data = (data - stats["mean"]) / stats["std"]
    else: 
        assert 0, f"Normalization type {normalize_type} not implemented"
    return data

def unnormalize(data, stats, normalize_type="normalize_feature"):     
    """Unnormalize data using statistics."""
    if 'normalize' in normalize_type:
        data = (data + 1) * (stats["max"] - stats["min"]) / 2 + stats["min"]
    elif 'meanstd' in normalize_type:
        data = data * stats["std"] + stats["mean"]
    else: 
        assert 0, f"Normalization type {normalize_type} not implemented"
    return data


def compute_stats(data, normalize_type="feature"):
    """Compute statistics on a list of numbers.

    Args:
        data (list): A list of numbers.

    Returns:
        dict: A dictionary containing the following statistics:
            - mean
            - median
            - min
            - max
            - range
            - variance
            - standard deviation
    """
    assert len(data.shape) in [2,3], "data must be 2D or 3D"
    stats = {
                "min": data.min(axis=0).values,
                "max": data.max(axis=0).values,
                "mean": data.mean(axis=0),
                "std": data.std(axis=0),
                "var": data.var(axis=0),
            }
    
    if normalize_type!= "feature" and ('latent' in normalize_type or 'node' in normalize_type):
        axis = 1 if 'node' in normalize_type else 2 #if  2: latent space dim, 1: nodes dim
        axis -= 1 # minus batch dim.
        stats["min"] = stats["min"].min(axis=axis).values
        stats["max"] = stats["max"].max(axis=axis).values
        stats["mean"] = stats["mean"].mean(axis=axis)
        stats["std"] = stats["std"].std(axis=axis)
        stats["var"] = stats["var"].var(axis=axis)
        for n in stats:
            stats[n] = stats[n].unsqueeze(axis).unsqueeze(0)
    return stats

