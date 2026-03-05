"""
Batch prediction API for exported RF toxicity models.

Usage:
    from analyses.utils.predict import ToxPredictor
    pred = ToxPredictor()
    results = pred.predict(["RNA1{...}$$$$V2.0", ...], species="mouse")
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from analyses.utils.helm import Helm
from analyses.utils.models import dinucleotide_features

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "data/models"


class ToxPredictor:
    """Load saved RF models and run batch toxicity predictions."""

    def __init__(self, model_dir: Path | None = None):
        model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR

        with open(model_dir / "metadata.json") as f:
            self._metadata = json.load(f)

        self._hepato_rf = load(model_dir / "hepatotox_alt.joblib")
        self._neuro_rf = load(model_dir / "neurotox_fob.joblib")

        self._hepato_meta = self._metadata["hepatotox_alt"]
        self._neuro_meta = self._metadata["neurotox_fob"]

        self._dinuc_names, self._dinuc_extract = dinucleotide_features()

    def predict(
        self,
        helm_strings: list[str],
        species: str = "mouse",
        num_doses: float | None = None,
        dosing_period_days: float | None = None,
    ) -> pd.DataFrame:
        """Batch predict hepatotoxicity and neurotoxicity.

        Args:
            helm_strings: HELM annotation strings.
            species: "mouse" or "rat" (sets species_rat feature).
            num_doses: Dosing covariate for hepatotox. None uses training median.
            dosing_period_days: Dosing covariate for hepatotox. None uses training median.

        Returns:
            DataFrame with columns: helm, p_hepatotoxic, hepatotoxic,
            p_neurotoxic, neurotoxic, valid.
        """
        n = len(helm_strings)
        species_rat = 1 if species == "rat" else 0

        if num_doses is None:
            num_doses = self._hepato_meta["default_num_doses"]
        if dosing_period_days is None:
            dosing_period_days = self._hepato_meta["default_dosing_period_days"]

        # Parse HELM and extract features
        valid = np.ones(n, dtype=bool)
        dinuc_rows = []
        for i, helm in enumerate(helm_strings):
            if not Helm.valid_chemistry(helm):
                valid[i] = False
                dinuc_rows.append({f: 0 for f in self._dinuc_names})
                continue
            parsed = Helm.parse(helm)
            if parsed is None:
                valid[i] = False
                dinuc_rows.append({f: 0 for f in self._dinuc_names})
                continue
            dinuc_rows.append(self._dinuc_extract(helm))

        dinuc_df = pd.DataFrame(dinuc_rows, columns=self._dinuc_names)

        # Build hepatotox features (128 dinuc + num_doses + dosing_period_days + species_rat)
        hepato_X = dinuc_df.copy()
        hepato_X["num_doses"] = num_doses
        hepato_X["dosing_period_days"] = dosing_period_days
        hepato_X["species_rat"] = species_rat
        hepato_X = hepato_X[self._hepato_meta["feature_names"]]

        # Build neurotox features (128 dinuc + species_rat)
        neuro_X = dinuc_df.copy()
        neuro_X["species_rat"] = species_rat
        neuro_X = neuro_X[self._neuro_meta["feature_names"]]

        # Predict
        p_hepato = self._hepato_rf.predict_proba(hepato_X)[:, 1]
        p_neuro = self._neuro_rf.predict_proba(neuro_X)[:, 1]

        # Threshold
        hepato_binary = (p_hepato > self._hepato_meta["threshold"]).astype(int)
        neuro_binary = (p_neuro > self._neuro_meta["threshold"]).astype(int)

        # NaN out invalid rows
        p_hepato = p_hepato.astype(float)
        p_neuro = p_neuro.astype(float)
        p_hepato[~valid] = np.nan
        p_neuro[~valid] = np.nan
        hepato_binary = hepato_binary.astype(float)
        neuro_binary = neuro_binary.astype(float)
        hepato_binary[~valid] = np.nan
        neuro_binary[~valid] = np.nan

        return pd.DataFrame({
            "helm": helm_strings,
            "p_hepatotoxic": p_hepato,
            "hepatotoxic": hepato_binary,
            "p_neurotoxic": p_neuro,
            "neurotoxic": neuro_binary,
            "valid": valid,
        })
