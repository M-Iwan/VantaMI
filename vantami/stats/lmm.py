from typing import Union, List, Optional
from dataclasses import dataclass

import polars as pl

from vantami.stats.utils import round_to_significant


@dataclass(frozen=True)
class EffectsConfig:
    """
    Default thresholds for checking fixed and random effects.
    """

    # Number of observations per level within fixed effect
    min_fixed: int = 10
    suff_fixed: int = 30
    adeq_fixed: int = 100

    # Fraction of all observations per level within fixed effect
    min_frac_fixed: float = 0.02
    suff_frac_fixed: float = 0.05
    adeq_frac_fixed: float = 0.10

    # Number of unique sources per random effect
    min_random: int = 5
    suff_random: int = 10
    adeq_random: int = 20


def check_effects(df: pl.DataFrame, fixed_effects: Union[str, List[str]], random_effects: Optional[Union[str, List[str]]],
                  config: EffectsConfig = EffectsConfig()):
    """
    Evaluate if categorical fixed effects are sufficiently populated and spread.

    Parameters
    ----------
    df: pl.DataFrame
        A Polars DataFrame.
    fixed_effects: Union[str, List[str]]
        Column(s) name(s) with fixed effects.
    random_effects: Optional[Union[str, List[str]]]
        Column(s) name(s) with random effects. If None, random-effect spread checks are skipped.
    config: EffectsConfig
        Thresholds for checking fixed and random effects.

    Returns
    -------
    level_results: pl.DataFrame
        A Polars DataFrame with analysis for individual levels within fixed effects.
    summary_results: pl.DataFrame
        A Polars DataFrame with summary of findings.
    """

    def classify_value(value: float, min_value: float, suff_value: float, adeq_value: float):
        """
        Assign a coverage category according to the supplied thresholds.
        """
        if value < min_value:
            return "Insufficient"
        elif value < suff_value:
            return "Minimal"
        elif value < adeq_value:
            return "Sufficient"
        return "Adequate"

    outcome_rank = {
        "Insufficient": 0,
        "Minimal": 1,
        "Sufficient": 2,
        "Adequate": 3,
    }

    n_total = len(df)
    if n_total == 0:
        raise ValueError("Passed an empty DataFrame")

    if isinstance(fixed_effects, str):
        fixed_effects = [fixed_effects]

    if random_effects is None:
        random_effects = []

    if isinstance(random_effects, str):
        random_effects = [random_effects]

    required_columns = set(fixed_effects + random_effects)
    if missing_columns := required_columns.difference(set(df.columns)):
        raise ValueError(
            f"Columns {missing_columns} are missing from the DataFrame"
        )

    fixed_dfs = []
    summary_rows = []

    for fix_ef in fixed_effects:
        fix_ef_n_none = float(df[fix_ef].is_null().sum())
        fix_ef_frac_unk = round_to_significant(fix_ef_n_none / n_total, 3)

        df_known = df.filter(pl.col(fix_ef).is_not_null())
        fix_ef_n_known = len(df_known)

        levels = (
            df_known[fix_ef].drop_nulls().unique().to_list()
            if fix_ef_n_known > 0 else []
        )

        level_rows = []
        valid_levels = []

        for level in levels:
            df_level = df_known.filter(pl.col(fix_ef) == level)
            n_level = len(df_level)

            level_fraction = round_to_significant(n_level / n_total, 3)

            random_counts = {
                f"n_{random_effect}": (df_level[random_effect].drop_nulls().n_unique())
                for random_effect in random_effects
            }

            n_obs_outcome = classify_value(
                value=n_level,
                min_value=config.min_fixed,
                suff_value=config.suff_fixed,
                adeq_value=config.adeq_fixed,
            )

            frac_outcome = classify_value(
                value=level_fraction,
                min_value=config.min_frac_fixed,
                suff_value=config.suff_frac_fixed,
                adeq_value=config.adeq_frac_fixed,
            )

            random_outcomes = {
                f"{random_effect}": classify_value(
                    value=random_count,
                    min_value=config.min_random,
                    suff_value=config.suff_random,
                    adeq_value=config.adeq_random,
                )
                for random_effect, random_count in random_counts.items()
            }

            checks = [
                ("n_obs", n_obs_outcome),
                ("frac", frac_outcome),
                *random_outcomes.items(),
            ]

            outcome = min(
                (check_outcome for _, check_outcome in checks),
                key=lambda check_outcome: outcome_rank[check_outcome],
            )

            limiting_checks = [
                check_name for check_name, check_outcome in checks if check_outcome == outcome
            ]

            if outcome == "Insufficient":
                reason = f"Failed: {'|'.join(limiting_checks)}"

            elif outcome == "Minimal":
                reason = f"Minimal: {'|'.join(limiting_checks)}"
                valid_levels.append(level)

            else:
                reason = ""
                valid_levels.append(level)

            level_rows.append(pl.DataFrame({
                    "fixed_effect": fix_ef,
                    "level": level,
                    "n_obs": n_level,
                    "frac": level_fraction,
                    **random_counts,
                    "n_obs_outcome": n_obs_outcome,
                    "frac_outcome": frac_outcome,
                    **random_outcomes,
                    "outcome": outcome,
                    "reason": reason,
                }))

        n_valid = len(valid_levels)

        if n_valid == 0:
            fix_ef_outcome = "Insufficient"
            fix_ef_reason = "No valid levels"

        elif n_valid == 1:
            fix_ef_outcome = "Insufficient"
            fix_ef_reason = "Only one valid level"

        else:
            fix_ef_outcome = "Valid"
            fix_ef_reason = "Valid"

        if level_rows:
            fixed_dfs.append(pl.concat(level_rows, how="diagonal_relaxed"))

        summary_rows.append(pl.DataFrame({
                "fixed_effect": fix_ef,
                "n_total": n_total,
                "n_known": fix_ef_n_known,
                "n_missing": fix_ef_n_none,
                "frac_missing": fix_ef_frac_unk,
                "n_levels": len(levels),
                "n_valid_levels": n_valid,
                "valid_levels": "|".join(map(str, valid_levels)),
                "outcome": fix_ef_outcome,
                "reason": fix_ef_reason,
            }))

    level_results = (
        pl.concat(fixed_dfs, how="diagonal_relaxed") if fixed_dfs else pl.DataFrame()
    )

    summary_results = (
        pl.concat(summary_rows, how="diagonal_relaxed") if summary_rows else pl.DataFrame()
    )

    return level_results, summary_results
