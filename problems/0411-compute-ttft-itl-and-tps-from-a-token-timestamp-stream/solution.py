
import torch

def compute_inference_metrics(timestamps: torch.Tensor) -> dict:
    ttft = timestamps[1] - timestamps[0]

    intervals = timestamps[2:] - timestamps[1:-1]
    itl = intervals.mean() if intervals.numel() > 0 else timestamps.new_tensor(0.0)

    tps = (timestamps.numel() - 1) / (
        timestamps[-1] - timestamps[0]
    )

    return {
        "ttft": ttft,
        "tps": tps,
        "itl": itl
    }
