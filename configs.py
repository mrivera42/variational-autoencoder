"""Experiment configurations.

Each entry is a dict consumed by `train.run_experiment(config)`. Each varies
one hyperparameter off the baseline so the effect is isolated.
"""

_BASE = {
    "batch_size": 128,
    "lr": 1e-3,
    "num_epochs": 10,
    "in_dim": 28 * 28,
    "trunk_dim": 400,
    "latent_dim": 20,
}


def _with(**overrides):
    cfg = dict(_BASE)
    cfg.update(overrides)
    return cfg


EXPERIMENTS = [
    _with(name="baseline_z20"),                       # canonical reference
    _with(name="latent_z10", latent_dim=10),          # smaller latent
    _with(name="latent_z50", latent_dim=50),          # larger latent
    _with(name="latent_z100", latent_dim=100),        # much larger latent
    _with(name="long_train_z20", num_epochs=30),      # more epochs
]