import itertools
from typing import Literal

import numpy as np
import numpy.typing as npt

import polars as pl

from scipy.stats import mannwhitneyu, wilcoxon, rankdata
from statsmodels.stats.multitest import multipletests

from vantami.stats.utils import round_to_significant


def mwu_rank_biserial(u_stat: float, n1: int, n2: int):
    """
    Rank biserial effect size for Mann-Whitney U-test. Positive value correspond to values
    from population 1 being larger than from population 2. Descriptions are assigned as follows:
        abs(r) < 0.1 -> Negligible
        abs(r) < 0.3 -> Small
        abs(r) < 0.5 -> Medium
        abs(r) >= 0.5 -> Large

    Parameters
    ----------
    u_stat: float
        Test statistic from scipy.stats.mannwhitneyu
    n1: int
        Size of population 1.
    n2: int
        Size of population 2.

    Returns
    -------
    r: float
        Rank biserial effect size
    description: str
        Description of effect size
    """

    r =  (2 * u_stat) / (n1 * n2) - 1

    if (r_abs := np.abs(r)) < 0.1:
        description = "Negligible"
    elif r_abs <= 0.3:
        description = "Small"
    elif r_abs <= 0.5:
        description = "Medium"
    else:
        description = "Large"

    return r, description


def mp_rank_biserial(values_1: npt.NDArray, values_2: npt.NDArray):
    """
    Rank biserial effect size for paired Wilcoxon signed-rank tests.
    Descriptions are assigned as follows:
        abs(r) < 0.1 -> Negligible
        abs(r) < 0.3 -> Small
        abs(r) < 0.5 -> Medium
        abs(r) >= 0.5 -> Large

    Parameters
    ----------
    values_1: npt.NDArray
        Paired values from population 1.
    values_2: npt.NDArray
        Paired values from population 2.

    Returns
    -------
    r: float
        Rank biserial effect size
    description: str
        Description of effect size
    """
    differences = values_1 - values_2
    differences = differences[differences != 0]

    ranks = rankdata(np.abs(differences))

    r_plus = np.sum(ranks[differences > 0])
    r_minus = np.sum(ranks[differences < 0])

    r = (r_plus - r_minus) / (r_plus + r_minus)

    if (r_abs := np.abs(r)) < 0.1:
        description = "Negligible"
    elif r_abs < 0.3:
        description = "Small"
    elif r_abs < 0.5:
        description = "Medium"
    else:
        description = "Large"

    return r, description


def mann_whitney_u_test(values_1: npt.NDArray, values_2: npt.NDArray, label_1: str, label_2: str,
                        min_n: int = 3, alpha: float = 0.05, alternative: str = "two-sided"):
    """
    Run a Mann-Whitney U test between two groups.

    Parameters
    ----------
    values_1: npt.NDArray
        Samples from population 1.
    values_2: npt.NDArray
        Samples from population 2.
    label_1: str
        Name for population 1.
    label_2: str
        Name for population 2.
    min_n: int, default=3
        Minimum sample size required per group.
    alpha: float, default=0.05
        Significance level.
    alternative: str, default="two-sided"
        Alternative hypothesis passed to scipy.stats.mannwhitneyu.
    """

    n1, n2 = len(values_1), len(values_2)
    null_hypothesis = f"{label_1} ~ {label_2}"
    low_n = (n1 < min_n) or (n2 < min_n)

    result = {
        "null_hypothesis": null_hypothesis,
        "label_1": label_1,
        "label_2": label_2,
        "n1": n1,
        "n2": n2,
        "mean_1": None,
        "mean_2": None,
        "median_1": None,
        "median_2": None,
        "u_stat": None,
        "p_value": None,
        "rbc": None,
        "effect_size": None,
        "low_n_flag": low_n,
        "outcome": None,
    }

    if low_n:
        print(f"Sample size too small for <{null_hypothesis}>. N1: {n1}, N2: {n2}")
        return result

    u_stat, p_value = mannwhitneyu(values_1, values_2, alternative=alternative)
    rbc, description = mwu_rank_biserial(u_stat, n1, n2)

    if p_value < alpha:
        if rbc > 0:
            relation = ">"
        elif rbc < 0:
            relation = "<"
        else:
            relation = "~"
    else:
        relation = "~"

    result.update(
        {
            "mean_1": round_to_significant(values_1.mean(), 5),
            "mean_2": round_to_significant(values_2.mean(), 5),
            "median_1": round_to_significant(np.median(values_1), 5),
            "median_2": round_to_significant(np.median(values_2), 5),
            "u_stat": u_stat,
            "p_value": p_value,
            "rbc": rbc,
            "effect_size": description,
            "outcome": f"{label_1} {relation} {label_2}",
        }
    )

    return result


def paired_wilcoxon_test(values_1: npt.NDArray, values_2: npt.NDArray, label_1: str, label_2: str,
                         min_n: int = 3, alpha: float = 0.05, alternative: str = "two-sided"):
    """
    Run a paired Wilcoxon signed-rank test between two groups.

    Parameters
    ----------
    values_1: npt.NDArray
        Paired values from population 1.
    values_2: npt.NDArray
        Paired values from population 2.
    label_1: str
        Name for population 1.
    label_2: str
        Name for population 2.
    min_n: int, default=3
        Minimum number of paired samples required.
    alpha: float, default=0.05
        Significance level.
    alternative: str, default="two-sided"
        Alternative hypothesis passed to scipy.stats.wilcoxon.

    Returns
    -------
    result: dict
        Wilcoxon signed-rank test results and descriptive statistics.
    """

    n1, n2 = len(values_1), len(values_2)
    null_hypothesis = f"{label_1} ~ {label_2}"
    low_n = (n1 < min_n) or (n2 < min_n)

    result = {
        "null_hypothesis": null_hypothesis,
        "label_1": label_1,
        "label_2": label_2,
        "n1": n1,
        "n2": n2,
        "mean_1": None,
        "mean_2": None,
        "median_1": None,
        "median_2": None,
        "w_stat": None,
        "p_value": None,
        "rbc": None,
        "effect_size": None,
        "low_n_flag": low_n,
        "outcome": None,
    }

    if low_n:
        print(f"Sample size too small for <{null_hypothesis}>. N1: {n1}, N2: {n2}")
        return result

    w_stat, p_value = wilcoxon(values_1, values_2, alternative=alternative)
    rbc, effect_size = mp_rank_biserial(values_1, values_2)

    if p_value < alpha:
        if rbc > 0:
            relation = ">"
        elif rbc < 0:
            relation = "<"
        else:
            relation = "~"
    else:
        relation = "~"

    result.update(
        {
            "mean_1": round_to_significant(values_1.mean(), 5),
            "mean_2": round_to_significant(values_2.mean(), 5),
            "median_1": round_to_significant(np.median(values_1), 5),
            "median_2": round_to_significant(np.median(values_2), 5),
            "w_stat": w_stat,
            "p_value": p_value,
            "rbc": rbc,
            "effect_size": effect_size,
            "outcome": f"{label_1} {relation} {label_2}",
        }
    )

    return result


def run_family_tests(df: pl.DataFrame, value_col: str, eval_cols: str | list[str], repeat_cols: str | list[str],
                     test: Literal["Mann-Whitney", "Paired Wilcoxon"], min_n: int = 3, alpha: float = 0.05,
                     alternative: str = "two-sided", adjustment_method: str = "fdr_bh"):
    """
    Run pairwise tests and adjust p-values within one comparison family.

    Parameters
    ----------
    df: pl.DataFrame
        DataFrame containing observations from one comparison family.
    value_col: str
        Name of the column containing observed values.
    eval_cols: str | list[str]
        Column name(s) whose values are compared pairwise.
    repeat_cols: str | list[str]
        Column name(s) identifying repeated observations.
    test: Literal["Mann-Whitney", "Paired Wilcoxon"]
        Statistical test to apply.
    min_n: int, default=3
        Minimum sample size required for a test.
    alpha: float, default=0.05
        Significance level.
    alternative: Literal["two-sided", "less", "greater"]
        Alternative hypothesis passed to the selected SciPy test.
    adjustment_method: str, default="fdr_bh"
        P-value adjustment method passed to `statsmodels.stats.multitest.multipletests`.

    Returns
    -------
    pl.DataFrame
        Test results with effect sizes, adjusted p-values, and rejection flags.

    Notes
    -----
    Last updated: vantami 0.6.1rc6
    """

    if isinstance(eval_cols, str):
        eval_cols = [eval_cols]

    if isinstance(repeat_cols, str):
        repeat_cols = [repeat_cols]

    required_cols = [value_col, *eval_cols, *repeat_cols]
    missing_cols = set(required_cols) - set(df.columns)

    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing_cols)}.")

    if test not in {"Mann-Whitney", "Paired Wilcoxon"}:
        raise ValueError("< test > must be 'Mann-Whitney' or 'Paired Wilcoxon'.")

    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("< alternative > must be 'two-sided', 'less', or 'greater'.")

    if min_n < 1:
        raise ValueError("< min_n > must be at least 1.")

    if not 0 < alpha < 1:
        raise ValueError("< alpha > must be between 0 and 1.")

    try:
        multipletests([1.0], alpha=alpha, method=adjustment_method)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid adjustment_method: {adjustment_method!r}.") from exc

    df = df.cast({
        value_col: pl.Float64,
        **{col: pl.String for col in eval_cols + repeat_cols},
    })

    test_fn = {
        "Mann-Whitney": mann_whitney_u_test,
        "Paired Wilcoxon": paired_wilcoxon_test,
    }[test]

    family_results = []

    for eval_col in eval_cols:
        eval_results = []
        eval_pairs = itertools.combinations(df[eval_col].drop_nulls().unique().to_list(), r=2)

        remaining_eval_cols = [col for col in eval_cols if col != eval_col]
        key_cols = remaining_eval_cols + repeat_cols

        for pair in eval_pairs:
            pair_df = df.filter(pl.col(eval_col).is_in(pair))

            pair_df = pair_df.with_columns(
                pl.struct(key_cols).map_elements(
                    lambda struct: ":".join(struct[key] for key in key_cols),
                    return_dtype=pl.String,
                ).alias("_pair_key")
            ).pivot(
                index="_pair_key",
                on=eval_col,
                values=value_col,
            )

            values_1 = pair_df[pair[0]].to_numpy().reshape(-1)
            values_2 = pair_df[pair[1]].to_numpy().reshape(-1)

            if test == "Paired Wilcoxon" and values_1.shape != values_2.shape:
                raise ValueError("Paired Wilcoxon test requires values_1 and values_2 to have equal lengths.")

            result = test_fn(
                values_1=values_1,
                values_2=values_2,
                label_1=pair[0],
                label_2=pair[1],
                min_n=min_n,
                alpha=alpha,
                alternative=alternative,
            )

            eval_results.append(pl.DataFrame(result))

        if eval_results:
            family_results.append(
                pl.concat(eval_results).with_columns(
                    pl.lit(eval_col).alias("Evaluation")
                )
            )

    if not family_results:
        raise ValueError("No pairwise comparisons could be generated for the supplied family.")

    results = pl.concat(family_results).with_row_index("_row_id")
    valid_results = results.filter(pl.col("p_value").is_not_null())

    if valid_results.height:
        reject_null, p_value_adjusted, _, _ = multipletests(
            valid_results["p_value"].to_numpy(),
            alpha=alpha,
            method=adjustment_method,
        )

        adjusted_results = valid_results.with_columns(
            pl.Series("p_value_adjusted", p_value_adjusted),
            pl.Series("reject_null", reject_null),
        ).select(
            "_row_id",
            "p_value_adjusted",
            "reject_null",
        )

        results = results.join(
            adjusted_results,
            on="_row_id",
            how="left",
        )
    else:
        results = results.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("p_value_adjusted"),
            pl.lit(False).alias("reject_null"),
        )

    return results.with_columns(
        pl.col("reject_null").fill_null(False),
        pl.when(pl.col("reject_null").fill_null(False))
        .then(pl.col("outcome"))
        .otherwise(pl.col("null_hypothesis"))
        .alias("outcome_adjusted"),
    ).drop("_row_id")


def run_multiple_family_tests(df: pl.DataFrame, value_col: str, eval_cols: str | list[str], family_cols: str | list[str],
                              repeat_cols: str | list[str], test: Literal["Mann-Whitney", "Paired Wilcoxon"], min_n: int = 3,
                              alpha: float = 0.05, alternative: str = "two-sided", adjustment_method: str = "fdr_bh"):
    """
    Run pairwise tests across multiple comparison families.

    Parameters
    ----------
    df: pl.DataFrame
        DataFrame containing observed values and comparison columns.
    value_col: str
        Name of the column containing observed values.
    eval_cols: str | list[str]
        Column name(s) whose values are compared pairwise.
    family_cols: str | list[str]
        Column name(s) defining multiple-testing families.
    repeat_cols: str | list[str]
        Column name(s) identifying repeated observations.
    test: Literal["Mann-Whitney", "Paired Wilcoxon"]
        Statistical test to apply.
    min_n: int, default=3
        Minimum sample size required for a test.
    alpha: float, default=0.05
        Significance level.
    alternative: Literal["two-sided", "less", "greater"]
        Alternative hypothesis passed to the selected SciPy test.
    adjustment_method: str, default="fdr_bh"
        P-value adjustment method passed to `statsmodels.stats.multitest.multipletests`

    Returns
    -------
    pl.DataFrame
        Family-wise test results with adjusted p-values and effect sizes.

    Notes
    -----
    Last updated: vantami 0.6.1rc6
    """

    if isinstance(family_cols, str):
        family_cols = [family_cols]

    if isinstance(eval_cols, str):
        eval_cols = [eval_cols]

    if isinstance(repeat_cols, str):
        repeat_cols = [repeat_cols]

    required_cols = [value_col, *eval_cols, *family_cols, *repeat_cols]
    missing_cols = set(required_cols) - set(df.columns)

    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing_cols)}.")

    if test not in {"Mann-Whitney", "Paired Wilcoxon"}:
        raise ValueError("< test > must be 'Mann-Whitney' or 'Paired Wilcoxon'.")

    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("< alternative > must be 'two-sided', 'less', or 'greater'.")

    if min_n < 1:
        raise ValueError("< min_n > must be at least 1.")

    if not 0 < alpha < 1:
        raise ValueError("< alpha > must be between 0 and 1.")

    try:
        multipletests([1.0], alpha=alpha, method=adjustment_method)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid adjustment_method: {adjustment_method!r}.") from exc

    df = df.cast({
        value_col: pl.Float64,
        **{col: pl.String for col in family_cols + eval_cols + repeat_cols},
    })

    family_values = [
        sorted(df[col].drop_nulls().unique().to_list()) for col in family_cols
    ]

    results = []

    for family in itertools.product(*family_values):
        family_filter = pl.lit(True)

        for family_col, value in zip(family_cols, family):
            family_filter &= pl.col(family_col) == value

        family_df = df.filter(family_filter)

        family_results = run_family_tests(
            df=family_df,
            value_col=value_col,
            eval_cols=eval_cols,
            repeat_cols=repeat_cols,
            test=test,
            min_n=min_n,
            alpha=alpha,
            alternative=alternative,
            adjustment_method=adjustment_method,
        ).with_columns(
            pl.lit(":".join(family)).alias("Family")
        )

        results.append(family_results)

    return pl.concat(results)
