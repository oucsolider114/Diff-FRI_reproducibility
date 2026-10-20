"""Seabed-parameter-free continuous modal-wavenumber estimators.

The public estimators accept observations and water-column metadata only.  The
multi-stage schedule is the one used by the mature stage-26 training backend,
but its objective is intentionally restricted to data fit plus row-group
sparsity.  Candidate selection is based only on learned coefficient activity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from mode_fri.mode_frinet import ModeFRINET


NUMBER_OF_ATOMS = 24
SEARCH_LOWER_FRACTION = 0.95
SEARCH_UPPER_FRACTION = 0.999999
CLUSTER_RADIUS = 0.0015
MINIMUM_MODE_SEPARATION = 0.004
GRADIENT_CLIP_NORM = 5.0


@dataclass(frozen=True)
class MultiStageSchedule:
    coefficient_learning_rate: float = 1.0e-2
    group_lambda_start: float = 1.0e-3
    group_lambda_final: float = 5.0e-3
    warmup_steps: int = 300
    stage1_steps: int = 500
    stage1_k_learning_rate: float = 5.0e-3
    stage2_steps: int = 1000
    stage2_k_learning_rate_start: float = 5.0e-3
    stage2_k_learning_rate_end: float = 1.0e-3
    stage3_steps: int = 500
    stage3_k_learning_rate: float = 5.0e-4
    stage3_active_atoms: int = 8


DIRECT_SCHEDULE = MultiStageSchedule()
LEGACY_STEPS = 2500
LEGACY_COEFFICIENT_LEARNING_RATE = 1.0e-2
LEGACY_WAVENUMBER_LEARNING_RATE = 3.0e-3
LEGACY_GROUP_LAMBDA = 5.0e-3


@dataclass(frozen=True)
class _ValidatedInputs:
    observations: np.ndarray
    receiver_indices: np.ndarray
    sound_speed_profile: np.ndarray
    dz: float
    omega: float
    k_min: float
    k_max: float
    number_of_modes: int
    seed: int


def _validate_inputs(
    Y: np.ndarray,
    K: int,
    frequency: float,
    array_geometry: np.ndarray,
    water_depth: float,
    SSP: np.ndarray,
    seed: int,
) -> _ValidatedInputs:
    observations = np.asarray(Y, dtype=np.complex128)
    if observations.ndim != 2 or min(observations.shape) <= 0:
        raise ValueError("Y must be a non-empty [receivers, snapshots] matrix")
    if not np.all(np.isfinite(observations.real)) or not np.all(
        np.isfinite(observations.imag)
    ):
        raise ValueError("Y contains non-finite values")

    number_of_modes = int(K)
    if number_of_modes <= 0 or number_of_modes > NUMBER_OF_ATOMS:
        raise ValueError("K must satisfy 1 <= K <= NUMBER_OF_ATOMS")

    frequency_hz = float(frequency)
    depth_m = float(water_depth)
    if frequency_hz <= 0.0 or depth_m <= 0.0:
        raise ValueError("frequency and water_depth must be positive")

    receiver_depths = np.asarray(array_geometry, dtype=np.float64)
    if receiver_depths.ndim != 1 or receiver_depths.size != observations.shape[0]:
        raise ValueError("array_geometry must contain one depth per row of Y")
    if not np.all(np.isfinite(receiver_depths)):
        raise ValueError("array_geometry contains non-finite values")

    sound_speed = np.asarray(SSP, dtype=np.float64)
    if sound_speed.ndim != 1 or sound_speed.size < 3:
        raise ValueError("SSP must be a one-dimensional profile with at least 3 points")
    if not np.all(np.isfinite(sound_speed)) or np.any(sound_speed <= 0.0):
        raise ValueError("SSP must contain finite positive sound speeds")

    dz = depth_m / float(sound_speed.size - 1)
    water_grid = np.linspace(0.0, depth_m, sound_speed.size, dtype=np.float64)
    receiver_indices = np.rint(receiver_depths / dz).astype(np.int64)
    if np.any(receiver_indices < 0) or np.any(receiver_indices >= sound_speed.size):
        raise ValueError("array_geometry contains depths outside the water column")
    if not np.allclose(
        water_grid[receiver_indices], receiver_depths, rtol=0.0, atol=max(1.0e-9, dz * 1.0e-7)
    ):
        raise ValueError("array_geometry must align with the supplied SSP depth grid")

    omega = 2.0 * math.pi * frequency_hz
    k_min = SEARCH_LOWER_FRACTION * omega / float(np.max(sound_speed))
    k_max = SEARCH_UPPER_FRACTION * omega / float(np.min(sound_speed))
    if not 0.0 < k_min < k_max:
        raise ValueError("water-column metadata produced an invalid wavenumber interval")

    return _ValidatedInputs(
        observations=observations.copy(),
        receiver_indices=receiver_indices,
        sound_speed_profile=sound_speed.copy(),
        dz=dz,
        omega=omega,
        k_min=k_min,
        k_max=k_max,
        number_of_modes=number_of_modes,
        seed=int(seed),
    )


def _seed_everything(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _build_model(inputs: _ValidatedInputs) -> ModeFRINET:
    sound_speed_tensor = torch.as_tensor(
        inputs.sound_speed_profile, dtype=torch.float64, device=torch.device("cpu")
    )
    return ModeFRINET(
        number_of_atoms=NUMBER_OF_ATOMS,
        number_of_snapshots=inputs.observations.shape[1],
        k_min=inputs.k_min,
        k_max=inputs.k_max,
        omega=inputs.omega,
        sound_speed_profile=sound_speed_tensor,
        dz=inputs.dz,
        random_seed=inputs.seed,
    ).to(torch.device("cpu"))


def _normalized_observations(inputs: _ValidatedInputs) -> torch.Tensor:
    observations = torch.as_tensor(
        inputs.observations, dtype=torch.complex128, device=torch.device("cpu")
    )
    scale = torch.sqrt(torch.mean(torch.abs(observations) ** 2)).clamp_min(1.0e-14)
    return observations / scale


def _data_group_loss(
    model: ModeFRINET,
    normalized_Y: torch.Tensor,
    receiver_indices: torch.Tensor,
    group_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, object]:
    output = model()
    reconstruction = output.reconstruction.index_select(0, receiver_indices)
    fit_loss = torch.mean(torch.abs(reconstruction - normalized_Y) ** 2)
    group_loss = torch.mean(output.activities) / math.sqrt(normalized_Y.shape[1])
    total_loss = fit_loss + float(group_lambda) * group_loss
    return total_loss, fit_loss, group_loss, output


def _activity_only_nms(
    wavenumbers: np.ndarray,
    activities: np.ndarray,
    number_to_select: int,
) -> np.ndarray:
    candidate_k = np.asarray(wavenumbers, dtype=np.float64)
    candidate_activity = np.asarray(activities, dtype=np.float64)
    if candidate_k.ndim != 1 or candidate_k.shape != candidate_activity.shape:
        raise ValueError("wavenumbers and activities must be equal-length vectors")
    if number_to_select <= 0:
        raise ValueError("number_to_select must be positive")

    scores = candidate_activity
    order = np.argsort(scores)[::-1]
    suppressed = np.zeros(candidate_k.size, dtype=bool)
    selected: list[float] = []

    for index in order:
        if suppressed[index]:
            continue
        center_k = float(candidate_k[index])
        local_mask = np.abs(candidate_k - center_k) <= CLUSTER_RADIUS
        local_k = candidate_k[local_mask]
        local_weights = candidate_activity[local_mask]
        weight_sum = float(np.sum(local_weights))
        if weight_sum <= 1.0e-30:
            selected_k = center_k
        else:
            selected_k = float(np.sum(local_k * local_weights) / weight_sum)
        selected.append(selected_k)
        suppressed[np.abs(candidate_k - center_k) < MINIMUM_MODE_SEPARATION] = True
        if len(selected) >= number_to_select:
            break

    if len(selected) < number_to_select:
        raise RuntimeError("activity-only NMS did not select enough separated candidates")
    return np.sort(np.asarray(selected, dtype=np.float64))[::-1]


def _train_multistage(inputs: _ValidatedInputs) -> tuple[np.ndarray, np.ndarray]:
    schedule = DIRECT_SCHEDULE
    _seed_everything(inputs.seed)
    model = _build_model(inputs)
    normalized_Y = _normalized_observations(inputs)
    receiver_indices = torch.as_tensor(inputs.receiver_indices, dtype=torch.long)

    model.k_raw.requires_grad_(False)
    warmup_optimizer = torch.optim.Adam(
        [model.coefficient_real, model.coefficient_imag],
        lr=schedule.coefficient_learning_rate,
    )
    for _ in range(schedule.warmup_steps):
        warmup_optimizer.zero_grad()
        loss, _, _, _ = _data_group_loss(
            model, normalized_Y, receiver_indices, schedule.group_lambda_start
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Direct-MS warmup produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [model.coefficient_real, model.coefficient_imag], GRADIENT_CLIP_NORM
        )
        warmup_optimizer.step()

    model.k_raw.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [
            {"params": [model.k_raw], "lr": schedule.stage1_k_learning_rate},
            {
                "params": [model.coefficient_real, model.coefficient_imag],
                "lr": schedule.coefficient_learning_rate,
            },
        ]
    )

    for _ in range(schedule.stage1_steps):
        optimizer.param_groups[0]["lr"] = schedule.stage1_k_learning_rate
        optimizer.zero_grad()
        loss, _, _, _ = _data_group_loss(
            model, normalized_Y, receiver_indices, schedule.group_lambda_start
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Direct-MS stage 1 produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()

    for local_step in range(schedule.stage2_steps):
        progress = 1.0 if schedule.stage2_steps <= 1 else local_step / (schedule.stage2_steps - 1)
        group_lambda = schedule.group_lambda_start + progress * (
            schedule.group_lambda_final - schedule.group_lambda_start
        )
        k_learning_rate = schedule.stage2_k_learning_rate_start + progress * (
            schedule.stage2_k_learning_rate_end - schedule.stage2_k_learning_rate_start
        )
        optimizer.param_groups[0]["lr"] = k_learning_rate
        optimizer.zero_grad()
        loss, _, _, _ = _data_group_loss(
            model, normalized_Y, receiver_indices, group_lambda
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Direct-MS stage 2 produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        stage3_output = model()
        number_active = min(schedule.stage3_active_atoms, stage3_output.activities.numel())
        active_indices = torch.topk(stage3_output.activities, k=number_active).indices
    model.train()
    k_gradient_mask = torch.zeros_like(model.k_raw)
    k_gradient_mask[active_indices] = 1.0
    optimizer.param_groups[0]["lr"] = schedule.stage3_k_learning_rate

    for _ in range(schedule.stage3_steps):
        optimizer.zero_grad()
        loss, _, _, _ = _data_group_loss(
            model, normalized_Y, receiver_indices, schedule.group_lambda_final
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Direct-MS stage 3 produced a non-finite loss")
        loss.backward()
        if model.k_raw.grad is not None:
            model.k_raw.grad.mul_(k_gradient_mask)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_output = model()
    return (
        final_output.wavenumbers.detach().cpu().numpy().astype(np.float64).copy(),
        final_output.activities.detach().cpu().numpy().astype(np.float64).copy(),
    )


def _train_legacy(inputs: _ValidatedInputs) -> tuple[np.ndarray, np.ndarray]:
    _seed_everything(inputs.seed)
    model = _build_model(inputs)
    normalized_Y = _normalized_observations(inputs)
    receiver_indices = torch.as_tensor(inputs.receiver_indices, dtype=torch.long)
    optimizer = torch.optim.Adam(
        [
            {"params": [model.k_raw], "lr": LEGACY_WAVENUMBER_LEARNING_RATE},
            {
                "params": [model.coefficient_real, model.coefficient_imag],
                "lr": LEGACY_COEFFICIENT_LEARNING_RATE,
            },
        ]
    )
    for _ in range(LEGACY_STEPS):
        optimizer.zero_grad()
        loss, _, _, _ = _data_group_loss(
            model, normalized_Y, receiver_indices, LEGACY_GROUP_LAMBDA
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Legacy continuous training produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_output = model()
    return (
        final_output.wavenumbers.detach().cpu().numpy().astype(np.float64).copy(),
        final_output.activities.detach().cpu().numpy().astype(np.float64).copy(),
    )


def estimate(Y, K, frequency, array_geometry, water_depth, SSP, seed) -> np.ndarray:
    """Estimate K horizontal modal wavenumbers from array observations."""
    inputs = _validate_inputs(Y, K, frequency, array_geometry, water_depth, SSP, seed)
    candidate_k, activities = _train_multistage(inputs)
    return _activity_only_nms(candidate_k, activities, inputs.number_of_modes)


def estimate_legacy(Y, K, frequency, array_geometry, water_depth, SSP, seed) -> np.ndarray:
    """Run the original one-stage continuous estimator without boundary loss."""
    inputs = _validate_inputs(Y, K, frequency, array_geometry, water_depth, SSP, seed)
    candidate_k, activities = _train_legacy(inputs)
    return _activity_only_nms(candidate_k, activities, inputs.number_of_modes)



def estimate_from_coarse(
    Y,
    K,
    frequency,
    array_geometry,
    water_depth,
    SSP,
    seed,
    initial_wavenumbers,
) -> np.ndarray:
    """Refine data-driven coarse wavenumbers without seabed parameters.

    The coarse estimates are used only as initialization.  The refinement
    objective contains data fitting only and does not use seabed parameters,
    physical roots, root-derived cells, or KRAKEN truth.
    """
    inputs = _validate_inputs(Y, K, frequency, array_geometry, water_depth, SSP, seed)

    initial_k = np.asarray(initial_wavenumbers, dtype=np.float64).reshape(-1)
    if initial_k.size != inputs.number_of_modes:
        raise ValueError("initial_wavenumbers must contain exactly K values")
    if not np.all(np.isfinite(initial_k)):
        raise ValueError("initial_wavenumbers contains non-finite values")

    _seed_everything(inputs.seed)

    sound_speed_tensor = torch.as_tensor(
        inputs.sound_speed_profile,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    model = ModeFRINET(
        number_of_atoms=inputs.number_of_modes,
        number_of_snapshots=inputs.observations.shape[1],
        k_min=inputs.k_min,
        k_max=inputs.k_max,
        omega=inputs.omega,
        sound_speed_profile=sound_speed_tensor,
        dz=inputs.dz,
        random_seed=inputs.seed,
        initial_wavenumbers=initial_k,
        initialization_bandwidth=0.0,
    ).to(torch.device("cpu"))

    normalized_Y = _normalized_observations(inputs)
    receiver_indices = torch.as_tensor(
        inputs.receiver_indices,
        dtype=torch.long,
        device=torch.device("cpu"),
    )

    # Stage 1: keep k fixed and fit the multi-snapshot complex coefficients.
    model.k_raw.requires_grad_(False)
    coefficient_optimizer = torch.optim.Adam(
        [model.coefficient_real, model.coefficient_imag],
        lr=1.0e-2,
    )
    for _ in range(100):
        coefficient_optimizer.zero_grad()
        output = model()
        reconstruction = output.reconstruction.index_select(0, receiver_indices)
        loss = torch.mean(torch.abs(reconstruction - normalized_Y) ** 2)
        if not torch.isfinite(loss):
            raise FloatingPointError("coarse-to-continuous warmup produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [model.coefficient_real, model.coefficient_imag],
            GRADIENT_CLIP_NORM,
        )
        coefficient_optimizer.step()

    # Stage 2: jointly refine k and C using data fit only.
    model.k_raw.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [
            {"params": [model.k_raw], "lr": 8.0e-5},
            {
                "params": [model.coefficient_real, model.coefficient_imag],
                "lr": 2.0e-3,
            },
        ]
    )
    for _ in range(400):
        optimizer.zero_grad()
        output = model()
        reconstruction = output.reconstruction.index_select(0, receiver_indices)
        loss = torch.mean(torch.abs(reconstruction - normalized_Y) ** 2)
        if not torch.isfinite(loss):
            raise FloatingPointError("continuous refinement produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        estimated_k = (
            model.get_wavenumbers()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    return np.sort(estimated_k)[::-1]
