"""Traversal-speed model for the gaze module's plan deadline.

Restored from git tag ``s14-variant-graveyard`` (2026-09-03 finalized cycle
design). The GAM's ROLE has changed: it is no longer a speed profile the MPCC
tracks (that variant stays pruned) — it predicts the local cursor speed
v(s) from which the gaze module derives the plan deadline
``t_plan = integral ds / v(s)`` over the lead. The MPCC itself remains
anchor-driven: cruise speed still emerges as lookahead / deadline.

Feature set v5 (fitted on the 10-participant per-sample batch,
eval/eval-gaze-lead/local_speed_law.py): clearance = W/2, |kappa| local,
anticipatory kappa (max |kappa| over the next 50 mm), and RUNWAY — the arc
distance to the next steering demand (nearest point ahead with
|kappa| > K_DEMAND_RAD_M, or the path end when none remains, the target
itself being a braking point) — a fourth spline term, monotone increasing.

Runway was added because a fixed 50 mm window cannot separate "straight for
the next 50 mm" (a corner-tunnel leg, humans ~0.08 m/s at W=10) from
"straight to the goal" (the straight-tunnel trials, humans ~0.14 m/s at
W=10): both populations shared identical v4 features, the fit predicted
their pooled mean, and the simulator crawled through straight tunnels at
corner-leg speed (eval-main-10p Steering trials 11-15, model ~1.7-1.9x
slower than human at W<=16.5 mm). Grouped-CV verdict (local_speed_law.py,
2026-09-03): leave-one-trial-out CV R2 0.336 with runway vs 0.324 for the
v4 set; replacing the windowed kappa instead of adding to it is worse
(0.317), so v5 keeps both anticipatory terms.

Fit target v6 (fix 2, 2026-09-03): the GAM is fitted on arc-binned PACE
(tau = occupancy seconds per metre, local_speed_law.py local_pace_samples)
as a conditional arithmetic mean (GammaGAM, log link), and predictions
return speed = 1/tau_hat. Rationale: the simulator consumes the model as a
time integral t_plan = integral ds / v = integral tau_hat ds, so the
correct target is E[time per metre | features] — the v4/v5 per-timestep
log-speed fit was a time-weighted geometric mean, which over-weights slow
phases (more samples per metre) and under-states bursty movement: at
W=10 mm straight, geo-mean 0.101 m/s vs the MT-consistent L/T 0.122 m/s.
Steady-speed regimes (curved tunnels) are barely affected, which is why
the bias went unnoticed until the straight-tunnel trials. Monotonicity
constraints are flipped accordingly (pace falls with clearance and runway,
rises with curvature).

The floor/ceil clamp costs ~0.04 CV R2 at any feature set, so the deadline
uses ``predict_speed_raw`` (unclamped) — ``compute_speed_profile`` keeps the
historical clamped behaviour for compatibility.
"""

from __future__ import annotations

import numpy as np
from typing import Protocol, Optional, Tuple, runtime_checkable


@runtime_checkable
class SpeedModel(Protocol):
    """Interface for speed adaptation models."""

    def compute_speed_profile(
        self,
        s_samples: np.ndarray,
        clearance: np.ndarray,
        kappa: np.ndarray,
        kappa_ahead: np.ndarray,
        runway: np.ndarray,
    ) -> np.ndarray:
        """Return desired speed (m/s) at each arc-length position.

        Args:
            s_samples:   Arc-length array (N,).
            clearance:   Local clearance at each sample (N,).
            kappa:       |curvature| at each sample (N,).
            kappa_ahead: max |curvature| over the next KAPPA_AHEAD_M (N,).
            runway:      arc distance to the next steering demand (N,).

        Returns:
            speed: Desired speed array (N,).
        """
        ...


# Additive offsets for log-transform (handles kappa≈0 on straight tunnels)
_CLEARANCE_EPS = 1e-6
_KAPPA_EPS = 1e-4
_DKAPPA_EPS = 1e-4
_RUNWAY_EPS = 1e-3
# Version 5: adds the runway feature (arc distance to the next steering
# demand) as a fourth spline term. Version 6: the fit target is arc-binned
# pace (conditional-mean seconds per metre; predictions return 1/tau_hat)
# instead of per-timestep log-speed. Older pickles are refused.
_GAM_VERSION = 6
# Trained clearance range (widths 10-50 mm -> clearance 5-25 mm). Inputs are
# clipped here before prediction so out-of-range widths (free space, the
# planner's 10 m drawing corridors) never ride the spline extrapolation.
CLEARANCE_TRAINED_MIN = 0.005
CLEARANCE_TRAINED_MAX = 0.025
# Forward window for the anticipatory curvature feature (m).
KAPPA_AHEAD_M = 0.05
# Curvature above which a point counts as a steering demand for the runway
# feature (rad/m; same floor local_speed_law.py uses inside its log — a 1 m
# turn radius, far above the numerical noise of straight segments).
K_DEMAND_RAD_M = 1.0
# Trained runway range (longest straight training tunnel, m); clipped like
# clearance so longer corridors never ride the spline extrapolation.
RUNWAY_TRAINED_MAX = 0.458


class GAMSpeedModel:
    """Learned speed model using Generalized Additive Model.

    Fits the conditional-mean pace with a log link (GammaGAM):
        E[tau] = exp(f1(log clearance) + f2(log kappa)
                     + f3(log kappa_ahead) + f4(log runway))
    where f1, f4 are monotonically decreasing (more room / more runway =
    faster = lower pace) and f2, f3 monotonically increasing. Predictions
    return speed = 1 / tau_hat.  Requires pyGAM (optional dependency).
    """

    def __init__(self, base_speed: float = 0.13,
                 floor: float = 0.05, ceil: float = 0.5):
        self.base_speed = base_speed
        self.floor = floor
        self.ceil = ceil
        self.gam = None
        self._is_fitted = False

    def fit(self, clearance: np.ndarray, kappa: np.ndarray,
            kappa_ahead: np.ndarray, runway: np.ndarray,
            pace: np.ndarray,
            lam_grid: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None):
        """Fit GAM from arc-binned human pace observations.

        Args:
            clearance:   Local clearance at each observation.
            kappa:       |curvature| at each observation.
            kappa_ahead: max |curvature| over the next KAPPA_AHEAD_M.
            runway:      arc distance to the next steering demand (m).
            pace:        Human pace tau (s/m) of the arc bin — occupancy
                         time / bin length; capped upstream at the
                         stationary threshold (1/V_MIN).
            lam_grid:  Smoothing parameter grid for gridsearch.
            sample_weight: Optional per-observation weights. Use these to
                de-bias duration sampling: observations arrive per timestep,
                so a slow round contributes proportionally more samples than
                a fast round of the same trial and drags the fit toward the
                slow tail unless rounds are re-weighted to contribute equally.
        """
        try:
            from pygam import GammaGAM, s as spline_term
        except ImportError:
            raise ImportError(
                "pyGAM is required for GAMSpeedModel. "
                "Install with: pip install pygam"
            )

        X = np.column_stack([
            np.log(np.maximum(clearance, _CLEARANCE_EPS)),
            np.log(np.asarray(kappa, dtype=float) + _KAPPA_EPS),
            np.log(np.asarray(kappa_ahead, dtype=float) + _DKAPPA_EPS),
            np.log(np.asarray(runway, dtype=float) + _RUNWAY_EPS),
        ])
        y = np.maximum(np.asarray(pace, dtype=float), 1e-3)

        n_clearance_levels = len(np.unique(np.round(X[:, 0], decimals=3)))
        n_cl_splines = min(8, max(4, n_clearance_levels))

        # GammaGAM with the default log link: fits E[tau | X] with the same
        # multiplicative additive structure the log-space LinearGAM had, but
        # mean-targeting. Constraints are in pace space (inverse of speed).
        gam = GammaGAM(
            spline_term(0, n_splines=n_cl_splines, constraints='monotonic_dec')
            + spline_term(1, n_splines=12, constraints='monotonic_inc')
            + spline_term(2, n_splines=12, constraints='monotonic_inc')
            + spline_term(3, n_splines=12, constraints='monotonic_dec'),
            fit_intercept=True,
        )

        if lam_grid is None:
            lam_grid = np.logspace(-1, 4, 11)
        gam.gridsearch(X, y, lam=lam_grid, weights=sample_weight)

        self.gam = gam
        self._is_fitted = True

    def compute_speed_profile(
        self,
        s_samples: np.ndarray,
        clearance: np.ndarray,
        kappa: np.ndarray,
        kappa_ahead: np.ndarray,
        runway: np.ndarray,
    ) -> np.ndarray:
        """Predict speed profile using fitted GAM (clamped legacy path)."""
        if not self._is_fitted:
            raise RuntimeError("GAMSpeedModel has not been fitted yet.")

        X = np.column_stack([
            np.log(np.maximum(clearance, _CLEARANCE_EPS)),
            np.log(np.asarray(kappa, dtype=float) + _KAPPA_EPS),
            np.log(np.asarray(kappa_ahead, dtype=float) + _DKAPPA_EPS),
            np.log(np.asarray(runway, dtype=float) + _RUNWAY_EPS),
        ])
        speed = 1.0 / np.maximum(self.gam.predict(X), 1e-6)
        return np.clip(speed, self.floor, self.ceil)

    def predict_speed_raw(self, clearance: np.ndarray, kappa: np.ndarray,
                          kappa_ahead: np.ndarray,
                          runway: np.ndarray) -> np.ndarray:
        """Unclamped speed prediction — the finalized-design path.

        Same features as ``compute_speed_profile`` but WITHOUT the floor/ceil
        clamp: the clamp costs ~0.04 CV R2 and its floor blocks the slow
        tail (corner near-stops). Clearance and runway are clipped to their
        trained ranges so free-space widths and over-long corridors never
        extrapolate the spline.
        """
        if not self._is_fitted:
            raise RuntimeError("GAMSpeedModel has not been fitted yet.")
        cl = np.clip(np.asarray(clearance, dtype=float),
                     CLEARANCE_TRAINED_MIN, CLEARANCE_TRAINED_MAX)
        rw = np.clip(np.asarray(runway, dtype=float),
                     0.0, RUNWAY_TRAINED_MAX)
        X = np.column_stack([
            np.log(np.maximum(cl, _CLEARANCE_EPS)),
            np.log(np.asarray(kappa, dtype=float) + _KAPPA_EPS),
            np.log(np.asarray(kappa_ahead, dtype=float) + _DKAPPA_EPS),
            np.log(rw + _RUNWAY_EPS),
        ])
        # the GAM predicts mean pace tau (s/m); the deadline integral
        # t_plan = integral ds / v then equals integral tau_hat ds exactly
        return 1.0 / np.maximum(self.gam.predict(X), 1e-6)

    def save(self, path: str):
        """Save fitted model to file."""
        import pickle
        data = {
            'gam': self.gam,
            'base_speed': self.base_speed,
            'floor': self.floor,
            'ceil': self.ceil,
            '_is_fitted': self._is_fitted,
            '_version': _GAM_VERSION,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'GAMSpeedModel':
        """Load fitted model from file."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        saved_version = data.get('_version', 1)
        if saved_version < _GAM_VERSION:
            # v4 changed the MEANING of the third feature (dkappa_ds ->
            # anticipatory kappa), so older artifacts are wrong, not merely
            # stale.
            raise ValueError(
                f"GAM model at {path} was fitted with feature transform v{saved_version} "
                f"(current: v{_GAM_VERSION}). Retrain "
                f"(eval/eval-gaze-lead/train_traversal_gam.py).")
        model = cls(
            base_speed=data['base_speed'],
            floor=data['floor'],
            ceil=data['ceil'],
        )
        model.gam = data['gam']
        model._is_fitted = data['_is_fitted']
        return model

    def plot_partial_effects(self, ax=None):
        """Plot the learned partial effect curves (3 main + interaction)."""
        if not self._is_fitted:
            raise RuntimeError("GAMSpeedModel has not been fitted yet.")
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        n_terms = len(self.gam.terms) - 1  # exclude intercept term
        if ax is None:
            fig, axes = plt.subplots(1, min(n_terms, 4), figsize=(4 * min(n_terms, 4), 4))
            if n_terms == 1:
                axes = [axes]
        else:
            axes = ax

        titles = ['Clearance', 'Curvature (kappa)', 'Anticipatory kappa', 'Runway to demand']
        for i in range(min(n_terms, len(axes))):
            a = axes[i]
            XX = self.gam.generate_X_grid(term=i)
            pdep = self.gam.partial_dependence(term=i, X=XX)
            if i < 4:
                a.plot(np.exp(XX[:, i]), np.exp(pdep))
            else:
                a.plot(np.exp(pdep))
            a.set_title(titles[i] if i < len(titles) else f'Term {i}')
            a.set_xlabel('Feature value')
            a.set_ylabel('Speed factor')

        if ax is None:
            plt.tight_layout()
            return fig
        return axes
