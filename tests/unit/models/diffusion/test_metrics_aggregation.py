import torch

from nemo_rl.models.diffusion.policy import aggregate_worker_metrics
from nemo_rl.models.diffusion.workers.diffusion_worker import accumulate_metrics


def test_accumulate_metrics_min_max_and_weighted_mean():
    acc: dict[str, float] = {}
    accumulate_metrics(
        acc, {"loss": torch.tensor(2.0), "ratio_min": 0.9, "ratio_max": 1.1}, 0.5
    )
    accumulate_metrics(
        acc, {"loss": torch.tensor(4.0), "ratio_min": 0.8, "ratio_max": 1.05}, 0.5
    )
    assert acc["loss"] == 3.0  # 加权平均
    assert acc["ratio_min"] == 0.8  # min，而非加权和
    assert acc["ratio_max"] == 1.1  # max


def test_aggregate_worker_metrics_min_max_and_mean():
    out = aggregate_worker_metrics(
        [
            {"loss": 1.0, "ratio_min": 0.9, "ratio_max": 1.2},
            {"loss": 3.0, "ratio_min": 0.7, "ratio_max": 1.0},
        ]
    )
    assert out == {"loss": 2.0, "ratio_min": 0.7, "ratio_max": 1.2}
