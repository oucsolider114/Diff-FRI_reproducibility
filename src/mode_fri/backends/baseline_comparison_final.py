# -*- coding: utf-8 -*-
"""
Final Baseline Comparison
======================================================================

Methods
----------------------------------------------------------------------
1. FD_SVD_PAPER
   Li & Wang, JASA Express Letters, 2021.
   Finite-difference VLA + CSDM/SVD + SSP perturbation correction.

2. GRID_512_GOMP
   Fixed-grid multi-snapshot group OMP + least-squares.
   No off-grid refinement.

3. MODE_FRINET
   Our continuous differentiable Mode-FRINET
   + mode-order cell beam selection
   + safe physics refinement.

Forward data
----------------------------------------------------------------------
KRAKEN .MOD mode shapes + moving-source propagation structure.

All three methods see the SAME noisy VLA pressure data.

Initial controlled comparison:
    target K = 3
    forward K = 3

Later, after this baseline is validated, an 8-mode forward field can be
introduced as model-order mismatch / interference robustness test.

======================================================================
Paper-compatible geometry
----------------------------------------------------------------------
H       = 40 m
f       = 300 Hz
Nz      = 40 receiver depths: 1,...,40 m
source  = 32.5 m
R0      = 4000 m
NR      = 64
L       = 2 * lambda_max
lambda_max = 849.2 m

The paper states NR = L / Delta-r, therefore:
    Delta-r = L / NR

FD-SVD uses:
    P  : receiver rows 2,...,39 -> 38 x NR
    PD : (row 1...38 + row 3...40)/2 -> 38 x NR

======================================================================
Important fairness rules
----------------------------------------------------------------------
- KRAKEN mode shapes are used ONLY for data generation.
- KRAKEN true k are used ONLY for evaluation.
- Mode-FRINET inverse physics remains FD/Robin.
- Grid baseline uses the same FD/Robin inverse dictionary.
- FD-SVD is implemented independently from Mode-FRINET.
- No true k is used for restart selection.
- Raw and refined Mode-FRINET results are reported separately.
"""

from __future__ import annotations

import csv
import importlib.util
import itertools
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# ======================================================================
# Paths
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

MULTIMODE_SCRIPT = (
    Path(__file__).resolve().parent
    / "multimode_mode_frinet_final.py"
)

KRAKEN_IO_SCRIPT = (
    Path(__file__).resolve().parent
    / "kraken_mod_independent_validation.py"
)

KRAKEN_MOD_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "24_kraken_reference_environment"
    / "kraken_case"
    / "teacher_paper_grad05.mod"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "baseline_comparison_final"
)

FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# Dynamic import
# ======================================================================
def load_module(
    path: Path,
    module_name: str,
):

    if not path.exists():
        raise FileNotFoundError(
            f"\n没有找到：\n{path}\n"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"无法加载：{path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[module_name] = module

    spec.loader.exec_module(
        module
    )

    return module


MULTI = load_module(
    MULTIMODE_SCRIPT,
    "multimode_for_final_baseline",
)

KRAKEN_IO = load_module(
    KRAKEN_IO_SCRIPT,
    "kraken_io_for_final_baseline",
)

EXP26 = MULTI.EXP26


# ======================================================================
# QUICK / FULL
# ======================================================================
QUICK_TEST = False


if QUICK_TEST:

    SNR_VALUES_DB = [
        5.0,
        15.0,
    ]

    NUMBER_OF_TRIALS = 2

    NUMBER_OF_RESTARTS = 2

    RUN_SUFFIX = "quick"

else:

    SNR_VALUES_DB = [
        -5.0,
        0.0,
        5.0,
        10.0,
        15.0,
        20.0,
        30.0,
    ]

    NUMBER_OF_TRIALS = 10

    NUMBER_OF_RESTARTS = 5

    RUN_SUFFIX = "full"


# ======================================================================
# Environment / paper settings
# ======================================================================
FREQUENCY_HZ = 300.0

OMEGA = (
    2.0
    * np.pi
    * FREQUENCY_HZ
)

WATER_DEPTH_M = 40.0

SOURCE_DEPTH_M = 32.5

REFERENCE_SOUND_SPEED = 1505.0

THERMOCLINE_TOP_M = 10.0

THERMOCLINE_BOTTOM_M = 20.0

DEEP_SOUND_SPEED = 1500.0


# ======================================================================
# Moving source geometry
#
# Paper:
# R0 = 4000 m
# lambda_max = 849.2 m
# L = 2 lambda_max
# NR = 64
# NR = L / delta_r
# ======================================================================
INITIAL_RANGE_M = 4000.0

MAX_MODAL_CYCLE_M = 849.2

RANGE_APERTURE_M = (
    2.0
    * MAX_MODAL_CYCLE_M
)

NUMBER_OF_RANGE_SAMPLES = 64

RANGE_STEP_M = (
    RANGE_APERTURE_M
    / NUMBER_OF_RANGE_SAMPLES
)


# ======================================================================
# Target problem
# ======================================================================
NUMBER_OF_TARGET_MODES = 3

NUMBER_OF_FORWARD_MODES = 3

SUCCESS_TOLERANCE = 1.5e-3

BASE_RANDOM_SEED = 20260910


# ======================================================================
# Mode-FRINET
# ======================================================================
NUMBER_OF_ATOMS = 24

TRAINING_K_MIN = 1.20

EXP26.NUMBER_OF_ATOMS = (
    NUMBER_OF_ATOMS
)

EXP26.NUMBER_OF_SNAPSHOTS = (
    NUMBER_OF_RANGE_SAMPLES
)

EXP26.K_MIN = (
    TRAINING_K_MIN
)


# ======================================================================
# Grid baseline
# ======================================================================
GRID_SIZE = 512

GRID_MIN = 1.20

GRID_MAX = float(
    EXP26.K_MAX
)

GRID_MIN_SEPARATION = 0.004


# ======================================================================
# Outputs
# ======================================================================
TRIAL_CSV = (
    OUTPUT_DIR
    / f"baseline_trials_{RUN_SUFFIX}.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / f"baseline_summary_{RUN_SUFFIX}.csv"
)

FRINET_RESTART_CSV = (
    OUTPUT_DIR
    / f"mode_frinet_restarts_{RUN_SUFFIX}.csv"
)

FD_DIAGNOSTIC_CSV = (
    OUTPUT_DIR
    / f"fd_svd_diagnostics_{RUN_SUFFIX}.csv"
)


# ======================================================================
# CSV tools
# ======================================================================
def save_csv(
    path: Path,
    rows: list[dict],
):

    if not rows:
        return

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def array_text(
    values,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    ).ravel()

    return ";".join(
        f"{value:.12f}"
        for value
        in values
    )


# ======================================================================
# SSP
# ======================================================================
def sound_speed_profile(
    depths: np.ndarray,
):

    z = np.asarray(
        depths,
        dtype=np.float64,
    )

    c = np.empty_like(
        z
    )

    mask_top = (
        z <= THERMOCLINE_TOP_M
    )

    mask_gradient = (
        (z > THERMOCLINE_TOP_M)
        &
        (z < THERMOCLINE_BOTTOM_M)
    )

    mask_deep = (
        z >= THERMOCLINE_BOTTOM_M
    )

    c[
        mask_top
    ] = REFERENCE_SOUND_SPEED

    c[
        mask_gradient
    ] = (
        REFERENCE_SOUND_SPEED
        +
        (
            DEEP_SOUND_SPEED
            - REFERENCE_SOUND_SPEED
        )
        *
        (
            z[
                mask_gradient
            ]
            - THERMOCLINE_TOP_M
        )
        /
        (
            THERMOCLINE_BOTTOM_M
            - THERMOCLINE_TOP_M
        )
    )

    c[
        mask_deep
    ] = DEEP_SOUND_SPEED

    return c


# ======================================================================
# Evaluation
# ======================================================================
def evaluate_k(
    estimate,
    truth,
):

    estimate = np.sort(
        np.asarray(
            estimate,
            dtype=np.float64,
        )
    )[::-1]

    truth = np.sort(
        np.asarray(
            truth,
            dtype=np.float64,
        )
    )[::-1]

    if (
        estimate.size
        != truth.size
    ):

        return {
            "success": False,
            "rmse": float("inf"),
            "max_error": float("inf"),
            "errors": np.full(
                truth.shape,
                np.inf,
            ),
        }

    errors = np.abs(
        estimate
        - truth
    )

    return {
        "success": bool(
            np.max(
                errors
            )
            <= SUCCESS_TOLERANCE
        ),

        "rmse": float(
            np.sqrt(
                np.mean(
                    errors ** 2
                )
            )
        ),

        "max_error": float(
            np.max(
                errors
            )
        ),

        "errors": (
            errors
        ),
    }


# ======================================================================
# Parse KRAKEN
# ======================================================================
def load_kraken_environment():

    mod = (
        KRAKEN_IO.parse_kraken_mod(
            KRAKEN_MOD_PATH
        )
    )

    receiver = (
        KRAKEN_IO.extract_integer_receiver_modes(
            mod
        )
    )

    receiver_depths = np.asarray(
        receiver[
            "receiver_depths"
        ],
        dtype=np.float64,
    )

    receiver_modes = np.asarray(
        receiver[
            "receiver_mode_matrix"
        ],
        dtype=np.complex128,
    )

    stored_depths = np.asarray(
        mod[
            "stored_depths"
        ],
        dtype=np.float64,
    )

    stored_modes = np.asarray(
        mod[
            "stored_mode_matrix"
        ],
        dtype=np.complex128,
    )

    complex_k = np.asarray(
        mod[
            "complex_wavenumbers"
        ],
        dtype=np.complex128,
    )

    matches = np.where(
        np.isclose(
            stored_depths,
            SOURCE_DEPTH_M,
            atol=1e-6,
            rtol=0.0,
        )
    )[0]

    if matches.size != 1:

        raise RuntimeError(
            "无法唯一定位KRAKEN source depth=32.5 m。"
        )

    source_index = int(
        matches[0]
    )

    source_modes = np.asarray(
        stored_modes[
            source_index,
            :
        ],
        dtype=np.complex128,
    )

    return {
        "receiver_depths": (
            receiver_depths
        ),

        "receiver_modes": (
            receiver_modes
        ),

        "source_modes": (
            source_modes
        ),

        "complex_k": (
            complex_k
        ),
    }


# ======================================================================
# Moving-source forward field
#
# Based on normal-mode structure in paper Eq.(1):
#
# p(z,r) proportional to
#
# sum_m phi_m(z_s) phi_m(z)
#       exp(i k_m r) / sqrt(k_m r)
#
# We use real(KRAKEN k) for phase.
#
# The overall constant is irrelevant because every estimator operates
# on the same data.
# ======================================================================
def generate_clean_moving_source_field(
    receiver_modes: np.ndarray,
    source_modes: np.ndarray,
    complex_k: np.ndarray,
):

    K = NUMBER_OF_FORWARD_MODES

    phi_receiver = np.asarray(
        receiver_modes[
            :,
            :K
        ],
        dtype=np.complex128,
    )

    phi_source = np.asarray(
        source_modes[
            :K
        ],
        dtype=np.complex128,
    )

    k = np.asarray(
        np.real(
            complex_k[
                :K
            ]
        ),
        dtype=np.float64,
    )

    ranges = (
        INITIAL_RANGE_M
        +
        np.arange(
            NUMBER_OF_RANGE_SAMPLES,
            dtype=np.float64,
        )
        * RANGE_STEP_M
    )

    coefficients = np.zeros(
        (
            K,
            NUMBER_OF_RANGE_SAMPLES,
        ),
        dtype=np.complex128,
    )

    for mode_index in range(
        K
    ):

        coefficients[
            mode_index,
            :
        ] = (
            phi_source[
                mode_index
            ]
            *
            np.exp(
                1j
                *
                k[
                    mode_index
                ]
                *
                ranges
            )
            /
            np.sqrt(
                k[
                    mode_index
                ]
                *
                ranges
            )
        )

    clean = (
        phi_receiver
        @ coefficients
    )

    return {
        "clean_receiver": (
            clean
        ),

        "ranges": (
            ranges
        ),

        "coefficients": (
            coefficients
        ),

        "true_k": (
            k[
                :NUMBER_OF_TARGET_MODES
            ].copy()
        ),
    }


# ======================================================================
# Same noisy realization for all methods
# ======================================================================
def add_noise(
    clean: np.ndarray,
    snr_db: float,
    random_seed: int,
):

    rng = np.random.default_rng(
        random_seed
    )

    noise = (
        rng.standard_normal(
            clean.shape
        )
        +
        1j
        *
        rng.standard_normal(
            clean.shape
        )
    ) / np.sqrt(2.0)

    clean_norm = float(
        np.linalg.norm(
            clean
        )
    )

    noise /= max(
        np.linalg.norm(
            noise
        ),
        1e-15,
    )

    target_noise_norm = (
        clean_norm
        /
        (
            10.0
            **
            (
                snr_db
                / 20.0
            )
        )
    )

    noise *= target_noise_norm

    noisy = (
        clean
        + noise
    )

    achieved_snr = float(
        20.0
        *
        np.log10(
            np.linalg.norm(
                clean
            )
            /
            max(
                np.linalg.norm(
                    noise
                ),
                1e-15,
            )
        )
    )

    return (
        noisy,
        achieved_snr,
    )


# ======================================================================
# Column normalization
#
# Paper notes geometric-spreading scalar can be removed by
# normalization at each range.
#
# We normalize ONCE and feed the SAME normalized matrix to all methods.
# ======================================================================
def normalize_snapshot_columns(
    matrix: np.ndarray,
):

    matrix = np.asarray(
        matrix,
        dtype=np.complex128,
    ).copy()

    norms = np.linalg.norm(
        matrix,
        axis=0,
    )

    norms = np.maximum(
        norms,
        1e-15,
    )

    return (
        matrix
        /
        norms[
            None,
            :
        ]
    )


# ======================================================================
# Adapt 40 physical receiver samples to Mode-FRINET inverse grid.
#
# If inverse model has 41 rows, z=0 pressure-release row is appended
# as exact physical boundary, not as an extra noisy hydrophone.
# ======================================================================
def adapt_to_inverse_grid(
    receiver_data: np.ndarray,
    probe_k: np.ndarray,
):

    probe = np.asarray(
        EXP26.build_mode_matrix(
            probe_k
        ),
        dtype=np.complex128,
    )

    if (
        probe.shape[0]
        ==
        receiver_data.shape[0]
    ):

        return (
            receiver_data.copy()
        )

    if (
        probe.shape[0]
        ==
        receiver_data.shape[0] + 1
    ):

        surface = np.zeros(
            (
                1,
                receiver_data.shape[1],
            ),
            dtype=np.complex128,
        )

        return np.vstack(
            [
                surface,
                receiver_data,
            ]
        )

    raise RuntimeError(
        "\nMode-FRINET inverse grid与40点VLA不兼容。\n"
        f"receiver data = {receiver_data.shape}\n"
        f"inverse probe = {probe.shape}\n"
    )


# ======================================================================
# FD-SVD baseline
# ======================================================================
def match_svd_modes(
    up: np.ndarray,
    ud: np.ndarray,
):

    K = up.shape[1]

    correlation = np.abs(
        up.conj().T
        @ ud
    )

    best_perm = None
    best_score = -np.inf

    for perm in itertools.permutations(
        range(K)
    ):

        score = sum(
            correlation[
                mode,
                perm[
                    mode
                ],
            ]
            for mode
            in range(K)
        )

        if score > best_score:

            best_score = (
                score
            )

            best_perm = (
                perm
            )

    return (
        np.asarray(
            best_perm,
            dtype=np.int64,
        ),
        correlation,
    )


def fd_svd_paper(
    receiver_data: np.ndarray,
    receiver_depths: np.ndarray,
):

    start = time.perf_counter()

    # --------------------------------------------------------------
    # Paper:
    #
    # 40 hydrophones at z=1,...,40
    #
    # P:
    # z=2,...,39
    #
    # PD:
    # [P(z-h)+P(z+h)]/2
    # --------------------------------------------------------------
    P = np.asarray(
        receiver_data[
            1:-1,
            :
        ],
        dtype=np.complex128,
    )

    PD = (
        np.asarray(
            receiver_data[
                :-2,
                :
            ],
            dtype=np.complex128,
        )
        +
        np.asarray(
            receiver_data[
                2:,
                :
            ],
            dtype=np.complex128,
        )
    ) / 2.0

    interior_depths = np.asarray(
        receiver_depths[
            1:-1
        ],
        dtype=np.float64,
    )

    h = float(
        receiver_depths[1]
        - receiver_depths[0]
    )

    NR = P.shape[1]

    # --------------------------------------------------------------
    # CSDM / FD-CSDM
    # --------------------------------------------------------------
    C = (
        P
        @ P.conj().T
        / NR
    )

    CD = (
        PD
        @ PD.conj().T
        / NR
    )

    # Hermitian eigendecomposition
    eigenvalues_p, eigenvectors_p = (
        np.linalg.eigh(
            C
        )
    )

    eigenvalues_d, eigenvectors_d = (
        np.linalg.eigh(
            CD
        )
    )

    order_p = np.argsort(
        eigenvalues_p
    )[::-1]

    order_d = np.argsort(
        eigenvalues_d
    )[::-1]

    eigenvalues_p = (
        np.real(
            eigenvalues_p[
                order_p
            ]
        )
    )

    eigenvalues_d = (
        np.real(
            eigenvalues_d[
                order_d
            ]
        )
    )

    eigenvectors_p = (
        eigenvectors_p[
            :,
            order_p
        ]
    )

    eigenvectors_d = (
        eigenvectors_d[
            :,
            order_d
        ]
    )

    K = NUMBER_OF_TARGET_MODES

    lambda_p = (
        eigenvalues_p[
            :K
        ]
    )

    lambda_d_candidate = (
        eigenvalues_d[
            :K
        ]
    )

    U_p = (
        eigenvectors_p[
            :,
            :K
        ]
    )

    U_d = (
        eigenvectors_d[
            :,
            :K
        ]
    )

    # --------------------------------------------------------------
    # Match same eigenvectors between C and CD.
    #
    # In exact theory they are the same U.
    # In noise / nonuniform SSP they are only approximate.
    # --------------------------------------------------------------
    permutation, correlation_matrix = (
        match_svd_modes(
            U_p,
            U_d,
        )
    )

    lambda_d = (
        lambda_d_candidate[
            permutation
        ]
    )

    matched_u_d = (
        U_d[
            :,
            permutation
        ]
    )

    matched_correlations = np.asarray(
        [
            abs(
                np.vdot(
                    U_p[
                        :,
                        m
                    ],
                    matched_u_d[
                        :,
                        m
                    ],
                )
            )
            for m
            in range(K)
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------------
    # Eq.(19):
    #
    # lambda_D / lambda_P
    # approximately equals |KD|^2
    #
    # KD = cos[h sqrt(k0^2-k_m^2)]
    # --------------------------------------------------------------
    ratio_squared = (
        lambda_d
        /
        np.maximum(
            lambda_p,
            1e-15,
        )
    )

    ratio_squared = np.clip(
        ratio_squared,
        0.0,
        1.0,
    )

    cosine_abs = np.sqrt(
        ratio_squared
    )

    cosine_abs = np.clip(
        cosine_abs,
        0.0,
        1.0,
    )

    vertical_phase = np.arccos(
        cosine_abs
    )

    kz_background = (
        vertical_phase
        / h
    )

    k0 = (
        OMEGA
        / REFERENCE_SOUND_SPEED
    )

    radicand = (
        k0 ** 2
        -
        kz_background ** 2
    )

    radicand = np.maximum(
        radicand,
        0.0,
    )

    background_k = np.sqrt(
        radicand
    )

    # --------------------------------------------------------------
    # Eq.(20) perturbation correction.
    #
    # Delta c(z) = c(z)-c0.
    #
    # delta k approximately:
    #
    # -(k0^2 / k_n)
    # integral [Delta c(z)/c0] |phi_n(z)|^2 dz
    #
    # U_p gives data-derived mode depth functions.
    #
    # Normalize mode energy before integral.
    # --------------------------------------------------------------
    c_z = (
        sound_speed_profile(
            interior_depths
        )
    )

    delta_c_ratio = (
        c_z
        -
        REFERENCE_SOUND_SPEED
    ) / REFERENCE_SOUND_SPEED

    corrected_k = np.zeros(
        K,
        dtype=np.float64,
    )

    perturbations = np.zeros(
        K,
        dtype=np.float64,
    )

    for mode_index in range(
        K
    ):

        phi = (
            U_p[
                :,
                mode_index
            ]
        )

        modal_energy = (
            np.abs(
                phi
            ) ** 2
        )

        energy_integral = float(
            np.trapezoid(
                modal_energy,
                interior_depths,
            )
        )

        if (
            energy_integral
            <= 1e-15
        ):

            corrected_k[
                mode_index
            ] = (
                background_k[
                    mode_index
                ]
            )

            continue

        modal_energy = (
            modal_energy
            / energy_integral
        )

        integral_value = float(
            np.trapezoid(
                delta_c_ratio
                *
                modal_energy,
                interior_depths,
            )
        )

        k_bg = max(
            float(
                background_k[
                    mode_index
                ]
            ),
            1e-12,
        )

        delta_k = (
            -
            (
                k0 ** 2
                / k_bg
            )
            *
            integral_value
        )

        perturbations[
            mode_index
        ] = delta_k

        corrected_k[
            mode_index
        ] = (
            k_bg
            + delta_k
        )

    # Mode order: descending horizontal k
    order = np.argsort(
        corrected_k
    )[::-1]

    corrected_k = (
        corrected_k[
            order
        ]
    )

    background_k = (
        background_k[
            order
        ]
    )

    perturbations = (
        perturbations[
            order
        ]
    )

    runtime = (
        time.perf_counter()
        - start
    )

    return {
        "estimated_k": (
            corrected_k
        ),

        "background_k": (
            background_k
        ),

        "perturbations": (
            perturbations
        ),

        "matched_correlations": (
            matched_correlations
        ),

        "ratio_squared": (
            ratio_squared
        ),

        "runtime": float(
            runtime
        ),

        "correlation_matrix": (
            correlation_matrix
        ),
    }


# ======================================================================
# Fixed-grid group OMP + LS
#
# This is intentionally kept grid-constrained:
# no continuous/off-grid refinement.
# ======================================================================
def grid_512_gomp(
    inverse_data: np.ndarray,
):

    start = time.perf_counter()

    grid = np.linspace(
        GRID_MAX,
        GRID_MIN,
        GRID_SIZE,
        dtype=np.float64,
    )

    dictionary = np.asarray(
        EXP26.build_mode_matrix(
            grid
        ),
        dtype=np.complex128,
    )

    # normalize atoms
    atom_norms = np.linalg.norm(
        dictionary,
        axis=0,
    )

    atom_norms = np.maximum(
        atom_norms,
        1e-15,
    )

    dictionary = (
        dictionary
        /
        atom_norms[
            None,
            :
        ]
    )

    residual = np.asarray(
        inverse_data,
        dtype=np.complex128,
    ).copy()

    selected_indices = []

    for _ in range(
        NUMBER_OF_TARGET_MODES
    ):

        correlations = (
            dictionary.conj().T
            @ residual
        )

        group_scores = np.linalg.norm(
            correlations,
            axis=1,
        )

        # remove already selected / too close candidates
        for grid_index in range(
            GRID_SIZE
        ):

            k_candidate = (
                grid[
                    grid_index
                ]
            )

            if any(
                abs(
                    k_candidate
                    -
                    grid[
                        old_index
                    ]
                )
                <
                GRID_MIN_SEPARATION

                for old_index
                in selected_indices
            ):

                group_scores[
                    grid_index
                ] = -np.inf

        best_index = int(
            np.argmax(
                group_scores
            )
        )

        selected_indices.append(
            best_index
        )

        selected_dictionary = (
            dictionary[
                :,
                selected_indices
            ]
        )

        coefficients, _, _, _ = (
            np.linalg.lstsq(
                selected_dictionary,
                inverse_data,
                rcond=None,
            )
        )

        residual = (
            inverse_data
            -
            selected_dictionary
            @ coefficients
        )

    estimated_k = np.sort(
        grid[
            selected_indices
        ]
    )[::-1]

    runtime = (
        time.perf_counter()
        - start
    )

    return {
        "estimated_k": (
            estimated_k
        ),

        "runtime": float(
            runtime
        ),
    }


# ======================================================================
# Mode-FRINET
# ======================================================================
def mode_frinet_estimate(
    inverse_data: np.ndarray,
    trial_seed: int,
    enable_refinement: bool = True,
):

    physical_roots = (
        MULTI.calculate_extended_roots()
    )

    target_roots = np.asarray(
        physical_roots[
            :NUMBER_OF_TARGET_MODES
        ],
        dtype=np.float64,
    )

    restart_results = []

    restart_rows = []

    total_start = (
        time.perf_counter()
    )

    for restart_index in range(
        NUMBER_OF_RESTARTS
    ):

        seed = (
            trial_seed
            +
            restart_index
            * 1009
        )

        restart_start = (
            time.perf_counter()
        )

        trained = (
            EXP26.train_one_restart(
                noisy_data=(
                    inverse_data
                ),
                random_seed=(
                    seed
                ),
            )
        )

        all_k = np.asarray(
            trained[
                "all_wavenumbers"
            ],
            dtype=np.float64,
        )

        activities = np.asarray(
            trained[
                "all_activities"
            ],
            dtype=np.float64,
        )

        selection = (
            MULTI.select_mode_order_beam(
                wavenumbers=(
                    all_k
                ),

                activities=(
                    activities
                ),

                noisy_data=(
                    inverse_data
                ),

                physical_roots=(
                    physical_roots
                ),

                number_of_modes=(
                    NUMBER_OF_TARGET_MODES
                ),

                training_k_min=(
                    TRAINING_K_MIN
                ),

                training_k_max=(
                    EXP26.K_MAX
                ),
            )
        )

        restart_runtime = (
            time.perf_counter()
            -
            restart_start
        )

        if (
            selection[
                "success"
            ]
        ):

            raw_k = np.asarray(
                selection[
                    "selected"
                ],
                dtype=np.float64,
            )

            score = float(
                selection[
                    "fit"
                ]
            )

        else:

            raw_k = np.asarray(
                [],
                dtype=np.float64,
            )

            score = float(
                "inf"
            )

        restart_results.append(
            {
                "restart": (
                    restart_index + 1
                ),

                "seed": (
                    seed
                ),

                "selector_success": bool(
                    selection[
                        "success"
                    ]
                ),

                "reason": (
                    selection[
                        "reason"
                    ]
                ),

                "raw_k": (
                    raw_k
                ),

                "score": (
                    score
                ),

                "runtime": (
                    restart_runtime
                ),
            }
        )

    valid = [
        index
        for index, result
        in enumerate(
            restart_results
        )
        if result[
            "selector_success"
        ]
        and
        np.isfinite(
            result[
                "score"
            ]
        )
    ]

    if not valid:

        total_runtime = (
            time.perf_counter()
            -
            total_start
        )

        return {
            "raw_k": np.asarray(
                [],
                dtype=np.float64,
            ),

            "refined_k": np.asarray(
                [],
                dtype=np.float64,
            ),

            "refine_accepted": False,

            "runtime": float(
                total_runtime
            ),

            "restart_results": (
                restart_results
            ),

            "best_restart": -1,
        }

    best_index = min(
        valid,
        key=lambda index:
        restart_results[
            index
        ][
            "score"
        ],
    )

    best = (
        restart_results[
            best_index
        ]
    )

    raw_k = np.asarray(
        best[
            "raw_k"
        ],
        dtype=np.float64,
    )

    if enable_refinement:
        refinement = MULTI.safe_refine(
            estimated=raw_k,
            physical_roots=target_roots,
            noisy_data=inverse_data,
        )
    else:
        refinement = {
            "refined": raw_k.copy(),
            "accepted": False,
        }

    refined_k = np.asarray(
        refinement[
            "refined"
        ],
        dtype=np.float64,
    )

    total_runtime = (
        time.perf_counter()
        -
        total_start
    )

    return {
        "raw_k": (
            raw_k
        ),

        "refined_k": (
            refined_k
        ),

        "refine_accepted": bool(
            refinement[
                "accepted"
            ]
        ),

        "runtime": float(
            total_runtime
        ),

        "restart_results": (
            restart_results
        ),

        "best_restart": (
            best_index + 1
        ),
    }


# ======================================================================
# Summary
# ======================================================================
METHODS = [
    "FD_SVD_PAPER",
    "GRID_512_GOMP",
    "MODE_FRINET_RAW",
    "MODE_FRINET_REFINED",
]


def summarize(
    rows: list[dict],
):

    summary = []

    for snr in SNR_VALUES_DB:

        for method in METHODS:

            subset = [
                row
                for row
                in rows
                if (
                    float(
                        row[
                            "snr_db"
                        ]
                    )
                    ==
                    float(
                        snr
                    )
                    and
                    row[
                        "method"
                    ]
                    ==
                    method
                )
            ]

            if not subset:
                continue

            rmse = np.asarray(
                [
                    float(
                        row[
                            "rmse"
                        ]
                    )
                    for row
                    in subset
                ],
                dtype=np.float64,
            )

            max_error = np.asarray(
                [
                    float(
                        row[
                            "max_error"
                        ]
                    )
                    for row
                    in subset
                ],
                dtype=np.float64,
            )

            runtime = np.asarray(
                [
                    float(
                        row[
                            "runtime_seconds"
                        ]
                    )
                    for row
                    in subset
                ],
                dtype=np.float64,
            )

            finite_rmse = (
                rmse[
                    np.isfinite(
                        rmse
                    )
                ]
            )

            finite_max = (
                max_error[
                    np.isfinite(
                        max_error
                    )
                ]
            )

            summary.append(
                {
                    "snr_db": (
                        snr
                    ),

                    "method": (
                        method
                    ),

                    "trials": (
                        len(
                            subset
                        )
                    ),

                    "success_rate": float(
                        np.mean(
                            [
                                int(
                                    row[
                                        "success"
                                    ]
                                )
                                for row
                                in subset
                            ]
                        )
                    ),

                    "rmse_median": (
                        float(
                            np.median(
                                finite_rmse
                            )
                        )
                        if finite_rmse.size
                        else float("nan")
                    ),

                    "max_error_median": (
                        float(
                            np.median(
                                finite_max
                            )
                        )
                        if finite_max.size
                        else float("nan")
                    ),

                    "runtime_seconds_mean": float(
                        np.mean(
                            runtime
                        )
                    ),
                }
            )

    return summary


# ======================================================================
# Plot
# ======================================================================
def plot_summary(
    summary_rows,
):

    fig, ax = plt.subplots(
        figsize=(
            9.0,
            5.8,
        )
    )

    for method in METHODS:

        subset = [
            row
            for row
            in summary_rows
            if row[
                "method"
            ]
            ==
            method
        ]

        subset.sort(
            key=lambda row:
            row[
                "snr_db"
            ]
        )

        ax.plot(
            [
                row[
                    "snr_db"
                ]
                for row
                in subset
            ],
            [
                100.0
                *
                row[
                    "success_rate"
                ]
                for row
                in subset
            ],
            marker="o",
            label=method,
        )

    ax.set_ylim(
        -5,
        105,
    )

    ax.set_xlabel(
        "SNR (dB)"
    )

    ax.set_ylabel(
        "Success rate (%)"
    )

    ax.set_title(
        "K=3 Moving-Source Baseline Comparison"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / (
            f"baseline_success_"
            f"{RUN_SUFFIX}.png"
        ),
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ======================================================================
# Main
# ======================================================================
def main():

    torch.set_default_dtype(
        torch.float64
    )

    print(
        "=" * 170
    )

    print(
        "FINAL BASELINE COMPARISON"
    )

    print(
        "=" * 170
    )

    print(
        f"run mode              = {RUN_SUFFIX}"
    )

    print(
        f"methods               = {METHODS}"
    )

    print(
        f"target modes          = {NUMBER_OF_TARGET_MODES}"
    )

    print(
        f"forward modes         = {NUMBER_OF_FORWARD_MODES}"
    )

    print(
        f"SNR                   = {SNR_VALUES_DB}"
    )

    print(
        f"trials / SNR          = {NUMBER_OF_TRIALS}"
    )

    print(
        f"Mode-FRINET restarts  = {NUMBER_OF_RESTARTS}"
    )

    print(
        f"Nz                    = 40"
    )

    print(
        f"NR                    = {NUMBER_OF_RANGE_SAMPLES}"
    )

    print(
        f"R0                    = {INITIAL_RANGE_M:.1f} m"
    )

    print(
        f"range aperture        = {RANGE_APERTURE_M:.1f} m"
    )

    print(
        f"delta-r               = {RANGE_STEP_M:.6f} m"
    )

    print(
        f"grid points           = {GRID_SIZE}"
    )

    print(
        "=" * 170
    )


    # ==================================================================
    # KRAKEN
    # ==================================================================
    kraken = (
        load_kraken_environment()
    )

    forward = (
        generate_clean_moving_source_field(
            receiver_modes=(
                kraken[
                    "receiver_modes"
                ]
            ),

            source_modes=(
                kraken[
                    "source_modes"
                ]
            ),

            complex_k=(
                kraken[
                    "complex_k"
                ]
            ),
        )
    )

    clean_receiver = np.asarray(
        forward[
            "clean_receiver"
        ],
        dtype=np.complex128,
    )

    true_k = np.asarray(
        forward[
            "true_k"
        ],
        dtype=np.float64,
    )

    receiver_depths = np.asarray(
        kraken[
            "receiver_depths"
        ],
        dtype=np.float64,
    )

    print(
        "\nKRAKEN truth:"
    )

    for index, value in enumerate(
        true_k,
        start=1,
    ):

        print(
            f"  k{index} = {value:.12f}"
        )

    print(
        f"\nclean receiver matrix = "
        f"{clean_receiver.shape}"
    )


    # ==================================================================
    # Monte Carlo
    # ==================================================================
    trial_rows = []

    frinet_restart_rows = []

    fd_diagnostic_rows = []


    for snr_db in SNR_VALUES_DB:

        print(
            "\n\n"
            + "=" * 170
        )

        print(
            f"SNR = {snr_db:+.1f} dB"
        )

        print(
            "=" * 170
        )

        EXP26.SNR_DB = float(
            snr_db
        )

        for trial in range(
            1,
            NUMBER_OF_TRIALS + 1,
        ):

            noise_seed = (
                BASE_RANDOM_SEED
                +
                int(
                    (
                        snr_db
                        + 20.0
                    )
                    * 1000
                )
                +
                trial
                * 100
            )

            noisy_receiver, achieved_snr = (
                add_noise(
                    clean=(
                        clean_receiver
                    ),

                    snr_db=(
                        snr_db
                    ),

                    random_seed=(
                        noise_seed
                    ),
                )
            )

            # Same preprocessing for every estimator
            common_receiver_data = (
                normalize_snapshot_columns(
                    noisy_receiver
                )
            )

            inverse_data = (
                adapt_to_inverse_grid(
                    receiver_data=(
                        common_receiver_data
                    ),

                    probe_k=(
                        true_k
                    ),
                )
            )

            print(
                f"\nTrial "
                f"{trial}/"
                f"{NUMBER_OF_TRIALS} | "
                f"achieved SNR="
                f"{achieved_snr:.6f} dB"
            )


            # ==========================================================
            # 1. FD-SVD paper baseline
            # ==========================================================
            fd = (
                fd_svd_paper(
                    receiver_data=(
                        common_receiver_data
                    ),

                    receiver_depths=(
                        receiver_depths
                    ),
                )
            )

            fd_eval = (
                evaluate_k(
                    fd[
                        "estimated_k"
                    ],
                    true_k,
                )
            )

            print(
                "  FD-SVD"
            )

            print(
                f"    k       = "
                f"{fd['estimated_k']}"
            )

            print(
                f"    success = "
                f"{fd_eval['success']}"
            )

            print(
                f"    RMSE    = "
                f"{fd_eval['rmse']:.6e}"
            )

            trial_rows.append(
                {
                    "snr_db": (
                        snr_db
                    ),

                    "trial": (
                        trial
                    ),

                    "method": (
                        "FD_SVD_PAPER"
                    ),

                    "success": int(
                        fd_eval[
                            "success"
                        ]
                    ),

                    "rmse": (
                        fd_eval[
                            "rmse"
                        ]
                    ),

                    "max_error": (
                        fd_eval[
                            "max_error"
                        ]
                    ),

                    "runtime_seconds": (
                        fd[
                            "runtime"
                        ]
                    ),

                    "estimated_k": (
                        array_text(
                            fd[
                                "estimated_k"
                            ]
                        )
                    ),
                }
            )

            for mode_index in range(
                NUMBER_OF_TARGET_MODES
            ):

                fd_diagnostic_rows.append(
                    {
                        "snr_db": (
                            snr_db
                        ),

                        "trial": (
                            trial
                        ),

                        "mode_index": (
                            mode_index + 1
                        ),

                        "background_k": float(
                            fd[
                                "background_k"
                            ][
                                mode_index
                            ]
                        ),

                        "perturbation": float(
                            fd[
                                "perturbations"
                            ][
                                mode_index
                            ]
                        ),

                        "final_k": float(
                            fd[
                                "estimated_k"
                            ][
                                mode_index
                            ]
                        ),

                        "matched_eigenvector_correlation": float(
                            fd[
                                "matched_correlations"
                            ][
                                mode_index
                            ]
                        ),

                        "singular_value_ratio_squared": float(
                            fd[
                                "ratio_squared"
                            ][
                                mode_index
                            ]
                        ),
                    }
                )


            # ==========================================================
            # 2. Grid baseline
            # ==========================================================
            grid = (
                grid_512_gomp(
                    inverse_data=(
                        inverse_data
                    )
                )
            )

            grid_eval = (
                evaluate_k(
                    grid[
                        "estimated_k"
                    ],
                    true_k,
                )
            )

            print(
                "  GRID-512-GOMP"
            )

            print(
                f"    k       = "
                f"{grid['estimated_k']}"
            )

            print(
                f"    success = "
                f"{grid_eval['success']}"
            )

            print(
                f"    RMSE    = "
                f"{grid_eval['rmse']:.6e}"
            )

            trial_rows.append(
                {
                    "snr_db": (
                        snr_db
                    ),

                    "trial": (
                        trial
                    ),

                    "method": (
                        "GRID_512_GOMP"
                    ),

                    "success": int(
                        grid_eval[
                            "success"
                        ]
                    ),

                    "rmse": (
                        grid_eval[
                            "rmse"
                        ]
                    ),

                    "max_error": (
                        grid_eval[
                            "max_error"
                        ]
                    ),

                    "runtime_seconds": (
                        grid[
                            "runtime"
                        ]
                    ),

                    "estimated_k": (
                        array_text(
                            grid[
                                "estimated_k"
                            ]
                        )
                    ),
                }
            )


            # ==========================================================
            # 3. Mode-FRINET
            # ==========================================================
            model_seed = (
                BASE_RANDOM_SEED
                +
                500000
                +
                trial
                * 10000
                +
                int(
                    (
                        snr_db
                        + 20
                    )
                    * 1000
                )
            )

            frinet = (
                mode_frinet_estimate(
                    inverse_data=(
                        inverse_data
                    ),

                    trial_seed=(
                        model_seed
                    ),
                )
            )

            raw_eval = (
                evaluate_k(
                    frinet[
                        "raw_k"
                    ],
                    true_k,
                )
            )

            refined_eval = (
                evaluate_k(
                    frinet[
                        "refined_k"
                    ],
                    true_k,
                )
            )

            print(
                "  MODE-FRINET"
            )

            print(
                f"    raw k       = "
                f"{frinet['raw_k']}"
            )

            print(
                f"    raw success = "
                f"{raw_eval['success']}"
            )

            print(
                f"    raw RMSE    = "
                f"{raw_eval['rmse']:.6e}"
            )

            print(
                f"    refined k   = "
                f"{frinet['refined_k']}"
            )

            print(
                f"    ref success = "
                f"{refined_eval['success']}"
            )

            print(
                f"    refine acc. = "
                f"{frinet['refine_accepted']}"
            )

            trial_rows.append(
                {
                    "snr_db": (
                        snr_db
                    ),

                    "trial": (
                        trial
                    ),

                    "method": (
                        "MODE_FRINET_RAW"
                    ),

                    "success": int(
                        raw_eval[
                            "success"
                        ]
                    ),

                    "rmse": (
                        raw_eval[
                            "rmse"
                        ]
                    ),

                    "max_error": (
                        raw_eval[
                            "max_error"
                        ]
                    ),

                    "runtime_seconds": (
                        frinet[
                            "runtime"
                        ]
                    ),

                    "estimated_k": (
                        array_text(
                            frinet[
                                "raw_k"
                            ]
                        )
                    ),
                }
            )

            trial_rows.append(
                {
                    "snr_db": (
                        snr_db
                    ),

                    "trial": (
                        trial
                    ),

                    "method": (
                        "MODE_FRINET_REFINED"
                    ),

                    "success": int(
                        refined_eval[
                            "success"
                        ]
                    ),

                    "rmse": (
                        refined_eval[
                            "rmse"
                        ]
                    ),

                    "max_error": (
                        refined_eval[
                            "max_error"
                        ]
                    ),

                    "runtime_seconds": (
                        frinet[
                            "runtime"
                        ]
                    ),

                    "estimated_k": (
                        array_text(
                            frinet[
                                "refined_k"
                            ]
                        )
                    ),
                }
            )

            for result in (
                frinet[
                    "restart_results"
                ]
            ):

                frinet_restart_rows.append(
                    {
                        "snr_db": (
                            snr_db
                        ),

                        "trial": (
                            trial
                        ),

                        "restart": (
                            result[
                                "restart"
                            ]
                        ),

                        "seed": (
                            result[
                                "seed"
                            ]
                        ),

                        "selector_success": int(
                            result[
                                "selector_success"
                            ]
                        ),

                        "reason": (
                            result[
                                "reason"
                            ]
                        ),

                        "selector_score": (
                            result[
                                "score"
                            ]
                        ),

                        "runtime_seconds": (
                            result[
                                "runtime"
                            ]
                        ),

                        "raw_k": (
                            array_text(
                                result[
                                    "raw_k"
                                ]
                            )
                        ),
                    }
                )


            # ==========================================================
            # Checkpoint
            # ==========================================================
            save_csv(
                TRIAL_CSV,
                trial_rows,
            )

            save_csv(
                FRINET_RESTART_CSV,
                frinet_restart_rows,
            )

            save_csv(
                FD_DIAGNOSTIC_CSV,
                fd_diagnostic_rows,
            )

            save_csv(
                SUMMARY_CSV,
                summarize(
                    trial_rows
                ),
            )


    # ==================================================================
    # Final
    # ==================================================================
    summary_rows = (
        summarize(
            trial_rows
        )
    )

    save_csv(
        SUMMARY_CSV,
        summary_rows,
    )

    plot_summary(
        summary_rows
    )


    print(
        "\n\n"
        + "=" * 175
    )

    print(
        "BASELINE SUMMARY"
    )

    print(
        "=" * 175
    )

    for row in summary_rows:

        print(
            f"SNR="
            f"{row['snr_db']:>5.1f} | "
            f"{row['method']:<22} | "
            f"success="
            f"{100*row['success_rate']:6.1f}% | "
            f"RMSE="
            f"{row['rmse_median']:.3e} | "
            f"max="
            f"{row['max_error_median']:.3e} | "
            f"time="
            f"{row['runtime_seconds_mean']:.3f}s"
        )


    print(
        "\nOutputs:"
    )

    print(
        SUMMARY_CSV
    )

    print(
        TRIAL_CSV
    )

    print(
        FRINET_RESTART_CSV
    )

    print(
        FD_DIAGNOSTIC_CSV
    )

    print(
        FIGURE_DIR
    )


if __name__ == "__main__":
    main()
