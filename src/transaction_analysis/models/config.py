from dataclasses import dataclass


@dataclass
class XGBConfig:
    n_estimators: int = 200
    max_depth: int = 7
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    scale_pos_weight: float = 10.0
    tree_method: str = "hist"
    eval_metric: str = "aucpr"
    random_state: int = 42
    n_threads: int = 1
    verbose: bool = True


@dataclass
class RFConfig:
    n_estimators: int = 500
    max_depth: int | None = None
    min_samples_leaf: int = 2
    class_weight: str = "balanced"
    n_jobs: int = 1
    random_state: int = 42
    verbose: bool = True
