"""Speed adaptation models for MPCC steering."""

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
        dkappa_ds: np.ndarray,
    ) -> np.ndarray:
        """Return desired speed (m/s) at each arc-length position.

        Args:
            s_samples:  Arc-length array (N,).
            clearance:  Local clearance at each sample (N,).
            kappa:      |curvature| at each sample (N,).
            dkappa_ds:  |curvature rate| at each sample (N,).

        Returns:
            speed: Desired speed array (N,).
        """
        ...


# Additive offsets for log-transform (handles kappa≈0 on straight tunnels)
_CLEARANCE_EPS = 1e-6
_KAPPA_EPS = 1e-4
_DKAPPA_EPS = 1e-4
_GAM_VERSION = 3  # Bump when feature transform changes (invalidates saved models)


class GAMSpeedModel:
    """Learned speed model using Generalized Additive Model.

    Fits in log-space:
        log(speed) = f1(log clearance) + f2(log kappa) + f3(log dkappa_ds)
    where f1 is monotonically increasing and f2, f3 are monotonically
    decreasing.  Requires pyGAM (optional dependency).
    """

    def __init__(self, base_speed: float = 0.13,
                 floor: float = 0.05, ceil: float = 0.5):
        self.base_speed = base_speed
        self.floor = floor
        self.ceil = ceil
        self.gam = None
        self._is_fitted = False

    def fit(self, clearance: np.ndarray, kappa: np.ndarray,
            dkappa_ds: np.ndarray, speeds: np.ndarray,
            lam_grid: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None):
        """Fit GAM from human speed observations.

        Args:
            clearance: Local clearance at each observation.
            kappa:     |curvature| at each observation.
            dkappa_ds: |curvature rate| at each observation.
            speeds:    Human speed at each observation.
            lam_grid:  Smoothing parameter grid for gridsearch.
            sample_weight: Optional per-observation weights. Use these to
                de-bias duration sampling: observations arrive per timestep,
                so a slow round contributes proportionally more samples than
                a fast round of the same trial and drags the fit toward the
                slow tail unless rounds are re-weighted to contribute equally.
        """
        try:
            from pygam import LinearGAM, s as spline_term
        except ImportError:
            raise ImportError(
                "pyGAM is required for GAMSpeedModel. "
                "Install with: pip install pygam"
            )

        X = np.column_stack([
            np.log(np.maximum(clearance, _CLEARANCE_EPS)),
            np.log(np.asarray(kappa, dtype=float) + _KAPPA_EPS),
            np.log(np.asarray(dkappa_ds, dtype=float) + _DKAPPA_EPS),
        ])
        y = np.log(np.maximum(speeds, 1e-6))

        n_clearance_levels = len(np.unique(np.round(X[:, 0], decimals=3)))
        n_cl_splines = min(8, max(4, n_clearance_levels))

        gam = LinearGAM(
            spline_term(0, n_splines=n_cl_splines, constraints='monotonic_inc')
            + spline_term(1, n_splines=12, constraints='monotonic_dec')
            + spline_term(2, n_splines=12, constraints='monotonic_dec'),
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
        dkappa_ds: np.ndarray,
    ) -> np.ndarray:
        """Predict speed profile using fitted GAM."""
        if not self._is_fitted:
            raise RuntimeError("GAMSpeedModel has not been fitted yet.")

        X = np.column_stack([
            np.log(np.maximum(clearance, _CLEARANCE_EPS)),
            np.log(np.asarray(kappa, dtype=float) + _KAPPA_EPS),
            np.log(np.asarray(dkappa_ds, dtype=float) + _DKAPPA_EPS),
        ])
        log_speed = self.gam.predict(X)
        speed = np.exp(log_speed)
        return np.clip(speed, self.floor, self.ceil)

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
        import warnings
        with open(path, 'rb') as f:
            data = pickle.load(f)
        saved_version = data.get('_version', 1)
        if saved_version < _GAM_VERSION:
            warnings.warn(
                f"GAM model at {path} was fitted with feature transform v{saved_version} "
                f"(current: v{_GAM_VERSION}). Retrain for correct results."
            )
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

        titles = ['Clearance', 'Curvature (kappa)', 'Curvature rate', 'Clearance × Curvature']
        for i in range(min(n_terms, len(axes))):
            a = axes[i]
            XX = self.gam.generate_X_grid(term=i)
            pdep = self.gam.partial_dependence(term=i, X=XX)
            if i < 3:
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
