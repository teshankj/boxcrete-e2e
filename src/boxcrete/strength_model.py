from __future__ import annotations

from collections.abc import Sequence

import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import (
    AffineInputTransform,
    AppendFeatures,
    ChainedInputTransform,
    Log10,
    Normalize,
)
from botorch.utils.constraints import LogTransformedInterval
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from boxcrete.features import (
    F5_ALLLOG_FEATURES,
    GATE_TAU,
    IDX,
    append_engineered_features_callable,
    augmented_bounds,
    max_scale_Y,
)

from boxcrete.kernels import (
    make_gated_strength_kernel_builder,
)

from boxcrete.likelihoods import (
    GatedGaussianLikelihood,
)

from boxcrete.utils import (
    derive_bounds_from_X,
)


CHAMPION_VARIANT = (
    "B''+F5_alllog+gated_t+"
    "gated_noise+maxscale_zeromean"
)


def get_v2_input_transform(
    d_in: int,
    bounds: Tensor,
    X_for_bounds: Tensor,
    feature_names: Sequence[str],
    log_time_offset: float = 1.0,
):
    """
    BOxCrete V2 input pipeline.

    1. Append 7 F5_alllog features
    2. Time -> Time + 1
    3. Time -> log10(Time + 1)
    4. Normalize every dimension except:
       - Material Source
       - Time
    """

    d_aug = d_in + len(feature_names)

    augmented = augmented_bounds(
        X_for_bounds,
        bounds,
        feature_names,
    )

    # ------------------------------------------------------------
    # 1. Append engineered features
    # ------------------------------------------------------------

    append_features = AppendFeatures(
        f=append_engineered_features_callable(
            feature_names
        ),
        indices=list(range(d_in)),
        transform_on_train=True,
        transform_on_eval=True,
        transform_on_fantasize=True,
    )

    # ------------------------------------------------------------
    # 2. Time + 1
    # ------------------------------------------------------------

    time_idx = IDX["time"]

    time_offset = AffineInputTransform(
        d=d_aug,
        coefficient=torch.ones(
            1,
            dtype=torch.float64,
        ),
        offset=torch.full(
            (1,),
            log_time_offset,
            dtype=torch.float64,
        ),
        indices=[time_idx],
        reverse=True,
    )

    # ------------------------------------------------------------
    # 3. log10(Time + 1)
    # ------------------------------------------------------------

    log_time = Log10(
        indices=[time_idx]
    )

    # ------------------------------------------------------------
    # 4. Normalize everything except:
    #
    #    Material Source
    #    Time
    # ------------------------------------------------------------

    source_idx = IDX["source"]

    normalize_indices = [
        i
        for i in range(d_aug)
        if i != time_idx
        and i != source_idx
    ]

    normalize = Normalize(
        d=d_aug,
        indices=torch.tensor(
            normalize_indices,
            dtype=torch.long,
        ),
        bounds=augmented,
    )

    return ChainedInputTransform(
        derive=append_features,
        log_offset=time_offset,
        log=log_time,
        normalize=normalize,
    )


def build_strength_v2(
    X: Tensor,
    Y: Tensor,
    X_bounds: Tensor | None = None,
    seed: int = 0,
):
    """
    Construct the BOxCrete V2 GP.

    X:
        [N, 10]

    Y:
        [N, 1]
    """

    if X.ndim != 2:
        raise ValueError(
            f"X must be [N,10], got {X.shape}"
        )

    if X.shape[-1] != 10:
        raise ValueError(
            "BOxCrete V2 expects exactly "
            f"10 raw features, got {X.shape[-1]}"
        )

    if Y.ndim != 2 or Y.shape[-1] != 1:
        raise ValueError(
            f"Y must have shape [N,1], got {Y.shape}"
        )

    # ------------------------------------------------------------
    # Float64
    # ------------------------------------------------------------

    X = X.to(torch.float64)
    Y = Y.to(torch.float64)

    if X_bounds is None:
        X_bounds = derive_bounds_from_X(X)

    X_bounds = X_bounds.to(torch.float64)

    torch.manual_seed(seed)

    # ------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------

    d_in = X.shape[-1]
    d_aug = d_in + len(
        F5_ALLLOG_FEATURES
    )

    # ------------------------------------------------------------
    # Y / ymax scaling
    # ------------------------------------------------------------

    Y_scaled, y_mean, y_std = max_scale_Y(Y)

    # ------------------------------------------------------------
    # Gated likelihood
    # ------------------------------------------------------------

    likelihood = GatedGaussianLikelihood(
        time_idx=IDX["time"],
        gate_tau=GATE_TAU,
        noise_constraint=LogTransformedInterval(
            1e-6,
            1.0,
            initial_value=1e-1,
        ),
    )

    likelihood.set_train_times(
        X[..., IDX["time"]]
    )

    # ------------------------------------------------------------
    # Research V2 kernel
    # ------------------------------------------------------------

    kernel_builder = (
        make_gated_strength_kernel_builder(
            gate_tau=GATE_TAU
        )
    )

    kernel = kernel_builder(
        d_aug
    )

    # ------------------------------------------------------------
    # Input transform
    # ------------------------------------------------------------

    input_transform = get_v2_input_transform(
        d_in=d_in,
        bounds=X_bounds,
        X_for_bounds=X,
        feature_names=F5_ALLLOG_FEATURES,
    )

    # ------------------------------------------------------------
    # Exact GP
    # ------------------------------------------------------------

    model = SingleTaskGP(
        train_X=X,
        train_Y=Y_scaled,
        covar_module=kernel,
        input_transform=input_transform,
        likelihood=likelihood,
        outcome_transform=None,
        mean_module=ZeroMean(),
    )

    # ------------------------------------------------------------
    # Re-register raw training times
    # ------------------------------------------------------------

    with torch.no_grad():
        train_times = (
            model.train_inputs[0]
            [..., IDX["time"]]
        )

    model.likelihood.set_train_times(
        train_times
    )

    # ------------------------------------------------------------
    # Store scaling information
    # ------------------------------------------------------------

    model.register_buffer(
        "_study_y_mean_buf",
        y_mean.squeeze().detach().clone(),
    )

    model.register_buffer(
        "_study_y_std_buf",
        y_std.squeeze().detach().clone(),
    )

    model._study_y_mean = (
        model._study_y_mean_buf
    )

    model._study_y_std = (
        model._study_y_std_buf
    )

    model._study_X_train_raw = (
        X.detach().clone()
    )

    return model, likelihood


def fit_strength_v2_mll(
    X: Tensor,
    Y: Tensor,
    X_bounds: Tensor | None = None,
    seed: int = 0,
    max_optimizer_iter: int | None = None,
):
    """
    MLL fit.

    This is useful for debugging and initialization.
    Production V2 should subsequently use
    block-LOO training.
    """

    model, likelihood = build_strength_v2(
        X=X,
        Y=Y,
        X_bounds=X_bounds,
        seed=seed,
    )

    model.train()
    likelihood.train()

    mll = ExactMarginalLogLikelihood(
        likelihood,
        model,
    )

    if max_optimizer_iter is None:

        fit_gpytorch_mll(mll)

    else:

        fit_gpytorch_mll(
            mll,
            optimizer_kwargs={
                "options": {
                    "maxiter": int(
                        max_optimizer_iter
                    )
                }
            },
        )

    return model, likelihood


def fit_strength_v2(
    X: Tensor,
    Y: Tensor,
    X_bounds: Tensor | None = None,
    seed: int = 0,
    mll_weight: float = 0.5,
    block_loo_max_iter: int = 150,
    block_loo_lr: float = 0.1,
    mll_warmup: bool = False,
    mll_max_iter: int = 20,
):
    """
    Full research-style BOxCrete V2 training.

    Default:
        construct fresh model
        optionally MLL warmup
        combined Block-LOO + MLL optimization

    The supplied research block_loo implementation defines the
    production combined objective.
    """

    model, likelihood = build_strength_v2(
        X=X,
        Y=Y,
        X_bounds=X_bounds,
        seed=seed,
    )

    n_real = X.shape[0]

    # ------------------------------------------------------------
    # Optional MLL warmup
    # ------------------------------------------------------------

    if mll_warmup:

        model.train()
        likelihood.train()

        mll = ExactMarginalLogLikelihood(
            likelihood,
            model,
        )

        fit_gpytorch_mll(
            mll,
            optimizer_kwargs={
                "options": {
                    "maxiter": int(
                        mll_max_iter
                    )
                }
            },
        )

    # ------------------------------------------------------------
    # Production block-LOO objective
    # ------------------------------------------------------------

    from boxcrete.block_loo import (
        train_block_loo,
    )

    block_loo_loss = train_block_loo(
        model,
        n_real=n_real,
        mll_weight=mll_weight,
        max_iter=block_loo_max_iter,
        lr=block_loo_lr,
    )

    model.eval()
    likelihood.eval()

    return (
        model,
        likelihood,
        block_loo_loss,
    )


def predict_strength(
    model,
    X,
):
    """
    Predict strength in original psi units.
    """

    X = X.to(
        dtype=torch.float64,
        device=next(model.parameters()).device,
    )

    model.eval()

    with torch.no_grad():

        posterior = model.posterior(X)

        mean_scaled = (
            posterior.mean
            .squeeze(-1)
        )

        variance_scaled = (
            posterior.variance
            .squeeze(-1)
            .clamp_min(0.0)
        )

        y_std = model._study_y_std

        mean = (
            mean_scaled
            * y_std
            + model._study_y_mean
        )

        std = (
            variance_scaled.sqrt()
            * y_std
        )

    return mean, std


__all__ = [
    "CHAMPION_VARIANT",
    "build_strength_v2",
    "fit_strength_v2_mll",
    "fit_strength_v2",
    "predict_strength",
]