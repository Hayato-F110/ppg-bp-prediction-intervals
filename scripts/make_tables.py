"""Supplementary tables of the paper, built from the result files so that no number is copied by hand.

Tables are numbered in the order of first citation in the main text. Table S9 (sensitivity analysis) is in the
text of the supplement. Descriptive information (exclusions, features, settings) is listed; comparative tables give
the exact values behind the figures of the main text.

usage: uv run python -X utf8 scripts/make_tables.py
input: result files under results/, including results/review*/ (scripts/explore_review*_analyses.py)
output: results/tables/supplement_tables.md
"""

import numpy as np
import pandas as pd

from ppg_bp.config import PROJECT_ROOT, load_config, load_settings
from ppg_bp.conformal.pipeline import band as label_bands
from ppg_bp.conformal.pipeline import band_labels, band_names
from ppg_bp.constants import COVERAGE_BIN_EDGES, FEW_PATIENTS, MAIN_LEVEL, SBP_BIN_EDGES
from ppg_bp.evaluation.plots import half_up

STRATA = load_settings("conformal")["strata"]
BANDS = band_names(STRATA["sbp"])
BAND_LABELS = dict(zip(BANDS, band_labels(STRATA["sbp"]), strict=True))  # as in the figures of the main text

MIN_KEPT_SHARE = 0.01  # withholding: values for fewer than 1% of segments kept are not shown
METHOD_NAMES = {
    "split": "Split conformal",
    "cqr": "CQR",
    "mondrian_pred": "Group-wise (predicted SBP)",
    "mondrian_sqi": "Group-wise (signal quality)",
    "mondrian_pred_sqi": "Group-wise (both)",
    "cqr_mondrian_pred_sqi": "CQR + group-wise (both)",
    "split_segment": "Split, equal weight per segment",
}
MODEL_NAMES = {"lightgbm": "Feature", "cnn": "Waveform"}
DBP_BANDS = band_names(STRATA["dbp"])
DBP_BAND_LABELS = dict(zip(DBP_BANDS, band_labels(STRATA["dbp"]), strict=True))
SUBGROUPS = {  # variable: (label, order of the groups)
    "within_subject": ("SBP relative to the patient's own median (mmHg)", ["<-20", "-20–<20", "≥20"]),
    "pulse_rate": ("Pulse rate (per minute)", ["<60", "60–<80", "80–<100", "≥100"]),
    "sqi": ("Signal-quality tertile", ["low", "middle", "high"]),
    "age": ("Age (years)", ["<40", "40–<60", "60–<75", "≥75"]),
    "sex": ("Sex", ["F", "M"]),
    "bmi": ("Body mass index", ["<18.5", "18.5–<25", "25–<30", "≥30"]),
}
DATASET_NAMES = {"vitaldb_test": "VitalDB test", "mimic3wdb": "MIMIC-III"}
COMPARISON_NAMES = {
    "cqr − split": "CQR − split",
    "mondrian_pred_sqi − split": "Group-wise (both) − split",
    "cqr_mondrian_pred_sqi − cqr": "CQR + group-wise − CQR",
}
MEASURE_NAMES = {
    "MAE (mmHg)": "MAE (mmHg)",
    "coverage split": "Coverage, split (%)",
    "coverage cqr": "Coverage, CQR (%)",
}
COMPARISON_STRATA = {"all": "all", "≥160": "SBP ≥ 160", "<90": "SBP < 90"}
REASONS = {
    "nan": "Missing samples",
    "abp_range": "Arterial pressure out of range",
    "flatline": "Constant signal for 1 s or longer",
    "few_beats": "Fewer than five beats",
    "pulse_rate": "Pulse rate out of range",
    "beat_irregular": "Unstable systolic pressure",
    "pulse_pressure": "Pulse pressure below 20 mmHg",
    "ok": "Kept",
}
FEATURES = [  # name, group, definition (src/ppg_bp/features/extract.py: segment_features and _morphology)
    ("n_beats", "Signal quality", "Number of detected beats in the segment"),
    ("b_a, c_a, d_a, e_a, f_a", "APG", "Amplitude of each APG wave divided by that of the a wave"),
    ("aging_index", "APG", "(b − c − d − e)/a"),
    ("apg_index", "APG", "(c + d − b)/a"),
    ("b_a_iqr", "APG", "Interquartile range of b/a over the beats (at least four beats)"),
    ("t_ab, t_ac, t_ad, t_ae, t_af", "APG", "Time from the a wave to each later wave (s)"),
    ("t_be, t_de", "APG", "Time from the b and d waves to the e wave (s)"),
    ("t_ae_rel", "APG", "Time from a to e divided by the beat duration"),
    (
        "crest_time, crest_time_rel",
        "Pulse shape",
        "Time from onset to systolic peak (s), and the same divided by the beat duration",
    ),
    (
        "upslope_time, max_slope",
        "Pulse shape",
        "Time from onset to the steepest upslope (s), and that slope divided by the pulse amplitude",
    ),
    (
        "sw25, sw50, sw75, dw25, dw50, dw75",
        "Pulse shape",
        "Width before (sw) and after (dw) the peak at 25%, 50% and 75% of the pulse amplitude (s)",
    ),
    (
        "width50_rel, dw_sw50",
        "Pulse shape",
        "Relative pulse width (width at 50% divided by the beat duration), and the width after the peak divided by "
        "that before it",
    ),
    ("area_ratio", "Pulse shape", "Area after the peak divided by the area before it"),
    (
        "t_peak_e, height_e",
        "Pulse shape",
        "Time from the systolic peak to the e wave (s), and the pulse height there relative to the peak",
    ),
    (
        "ibi_median, ibi_sd, ibi_rmssd, ibi_cv",
        "Rhythm",
        "Median, SD, root mean square of successive differences, and coefficient of variation of the pulse interval",
    ),
    ("sqi_template", "Signal quality", "Mean correlation of each pulse with the average pulse of the segment"),
    ("sqi_skewness, sqi_kurtosis", "Signal quality", "Skewness and kurtosis of the filtered PPG"),
    (
        "frac_e_found, frac_cd_found, frac_cd_distinct",
        "Signal quality",
        "Share of beats in which the e wave, the c and d waves, and distinct c and d waves were found",
    ),
]


def pct(x, digits=1):
    return half_up(100 * x, digits)


def ci(row):
    return f"{pct(row.coverage)} ({pct(row.coverage_lo)}–{pct(row.coverage_hi)})"


def markdown(table: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join(lines)


def strata_table(coverage, variable, order=None, model="lightgbm", methods=("split", "cqr")):
    part = coverage[
        (coverage.target == "sbp")
        & (coverage.level == MAIN_LEVEL)
        & (coverage.model == model)
        & (coverage.variable == variable)
    ]
    rows = []
    for stratum in order or sorted(part.stratum.unique()):
        row = {"Group": stratum}
        for method in methods:
            r = part[(part.method == method) & (part.stratum == stratum)]
            if r.empty:
                continue
            r = r.iloc[0]
            row["Patients"] = int(r.subjects)
            row[f"{METHOD_NAMES[method]} (95% CI)"] = ci(r) + (" †" if r.subjects < FEW_PATIENTS else "")
            row[f"{METHOD_NAMES[method]}: above / below"] = f"{pct(r.miss_above)} / {pct(r.miss_below)}"
        rows.append(row)
    return pd.DataFrame(rows)


def settings_table():
    feature = load_settings("feature_models")
    wave = load_settings("waveform_model")
    lgb = feature["lightgbm"]
    rows = [
        (
            "Both",
            "Training data",
            f"At most {feature['max_segments_per_subject']} random segments per training patient; "
            f"{100 * feature['validation_fraction']:.0f}% of the training patients held out for model selection",
        ),
        ("Both", "Outputs", "Mean, and quantiles " + ", ".join(f"{q:g}" for q in wave["quantiles"])),
        (
            "Feature model",
            "Algorithm",
            "LightGBM gradient-boosted trees; one model for the mean (squared error) and one per quantile (quantile "
            "loss)",
        ),
        (
            "Feature model",
            "Settings",
            f"Learning rate {lgb['learning_rate']}; up to {lgb['n_estimators']} trees, early stopping after "
            f"{lgb['early_stopping_rounds']} rounds; "
            f"{lgb['num_leaves']} leaves; at least {lgb['min_child_samples']} segments per leaf; row and feature "
            "subsampling "
            f"{lgb['subsample']} and {lgb['colsample_bytree']}; L2 penalty {lgb['reg_lambda']}",
        ),
        ("Feature model", "Missing features", "Handled by LightGBM without imputation"),
        ("Waveform model", "Input", "Raw 10-s PPG at 125 Hz, standardised per segment"),
        (
            "Waveform model",
            "Architecture",
            f"1-D convolution (15 taps, stride 2); four stages of two residual blocks (kernel 7) with "
            f"{', '.join(map(str, wave['channels']))} channels; global average pooling; linear output; about 450,000 "
            "parameters",
        ),
        (
            "Waveform model",
            "Loss",
            "Squared error for the mean plus pinball loss for the quantiles, on standardised targets",
        ),
        (
            "Waveform model",
            "Training",
            f"AdamW, learning rate {wave['learning_rate']} with a one-cycle schedule, weight decay "
            f"{wave['weight_decay']}, batch {wave['batch_size']}, "
            f"{wave['segments_per_epoch']:,} random segments per epoch, at most {wave['max_epochs']} epochs, "
            f"early stopping after {wave['patience']} epochs without improvement",
        ),
        ("Both", "Crossing quantiles", "Lower and upper quantiles swapped when they crossed"),
    ]
    return pd.DataFrame(rows, columns=["Model", "Item", "Setting"])


def overall_rows(coverage, dataset, targets):
    rows = []
    for target in targets:
        for model, model_name in MODEL_NAMES.items():
            for method, method_name in METHOD_NAMES.items():
                part = coverage[
                    (coverage.variable == "all")
                    & (coverage.target == target)
                    & (coverage.model == model)
                    & (coverage.method == method)
                ].set_index("level")
                if part.empty:
                    continue
                row = {
                    "Dataset": dataset,
                    "Target": target.upper(),
                    "Model": model_name,
                    "Method": method_name,
                }
                for level in (0.8, 0.9, 0.95):
                    row[f"Nominal {100 * level:.0f}% (95% CI)"] = ci(part.loc[level])
                row["Median width at 90% (mmHg)"] = half_up(part.loc[0.9].width_median, 1)
                row["Patients below 70% at 90% (%)"] = pct(part.loc[0.9].subjects_below_minus20)
                rows.append(row)
    return rows


def band_rows(coverage, dataset, model, methods, target="sbp", bands=BANDS, labels=BAND_LABELS):
    """Coverage (95% CI) by band of true pressure, after a first row with the number of patients per band."""
    rows = []
    for method in methods:
        part = coverage[
            (coverage.target == target)
            & (coverage.level == MAIN_LEVEL)
            & (coverage.model == model)
            & (coverage.method == method)
            & (coverage.variable == f"true_{target}")
        ].set_index("stratum")
        if part.empty:
            continue
        if not rows:
            rows.append(
                {
                    "Dataset, model, method": f"{dataset}: patients",
                    **{labels[b]: f"{int(part.loc[b].subjects):,}" for b in bands},
                }
            )
        row = {
            "Dataset, model, method": f"{dataset}: {MODEL_NAMES[model].lower()} model, {METHOD_NAMES[method]}"
        }  # one column, to avoid wrapping in narrow columns
        for band in bands:
            r = part.loc[band]
            row[labels[band]] = ci(r) + (" †" if r.subjects < FEW_PATIENTS else "")
        rows.append(row)
    return rows


def dagger_note(rows) -> list:
    """Add the note only when a dagger is used."""
    used = any("†" in str(v) for row in rows for v in row.values())
    return ["† Fewer than 30 patients; reported for reference only.", ""] if used else []


def section(title: str, note: str | None, table: pd.DataFrame) -> list:
    """Lines of one table: title, optional note, the table and a blank line."""
    lines = [title, ""]
    if note:
        lines += [note, ""]
    return [*lines, markdown(table), ""]


def table_s1(processed) -> list:
    """S1: segments kept and excluded, by the first rule failed."""
    summary = pd.read_csv(processed / "vitaldb" / "case_summary.csv")
    mimic = pd.read_csv(processed / "mimic3wdb" / "subject_summary.csv")
    cases = pd.read_csv(processed / "vitaldb" / "cases.csv", usecols=["caseid", "subjectid"])
    kept = cases[cases.caseid.isin(summary.loc[summary.ok > 0, "caseid"])]
    patients_all = cases.subjectid.nunique()
    patients_lost = patients_all - kept.subjectid.nunique()
    rows = []
    for key, label in REASONS.items():
        v, m = summary[key].sum(), mimic[key].sum()
        rows.append(
            {
                "Outcome": label,
                "VitalDB segments": f"{int(v):,}",
                "VitalDB %": pct(v / summary.n_segments.sum()),
                "MIMIC-III segments": f"{int(m):,}",
                "MIMIC-III %": pct(m / mimic.n_segments.sum()),
            }
        )
    return section(
        "## Table S1. Segments kept and excluded",
        "Rules were applied in the order shown; a segment is counted under the first rule that it failed. "
        f"In VitalDB, {patients_lost} of {patients_all:,} patients had no segment left.",
        pd.DataFrame(rows),
    )


def tables_s2_s3() -> list:
    """S2: the features; S3: the model settings."""
    return [
        *section(
            "## Table S2. The 42 features of the feature model",
            "Beat-level values are summarised by their median over the beats of a 10-s segment. "
            "Rhythm and signal-quality features describe the whole segment.",
            pd.DataFrame(FEATURES, columns=["Feature", "Group", "Definition"]),
        ),
        *section("## Table S3. Model settings", None, settings_table()),
    ]


def table_s4(results) -> list:
    """S4: accuracy of the point estimate in the test patients (explore_review_analyses.py)."""
    acc = pd.read_csv(results / "review" / "test_accuracy.csv")
    rows = [
        {
            "Target": r.target.upper(),
            "Model": r.predictor,
            "MAE": half_up(r.mae, 1),
            "Mean error ± SD": f"{'+' if r.me > 0 else '−' if r.me < 0 else ''}{half_up(abs(r.me), 1)} ± "
            f"{half_up(r.sd, 1)}",
            "Within 5 / 10 / 15 mmHg (%)": f"{half_up(r.within5, 0)} / {half_up(r.within10, 0)} / "
            f"{half_up(r.within15, 0)}",
        }
        for r in acc.itertuples()
    ]
    return section(
        f"## Table S4. Accuracy of the point estimate, {int(acc.patients.iloc[0])} VitalDB test patients "
        f"({int(acc.n.iloc[0]):,} segments), PPG-only inputs (mmHg)",
        "Error is the estimate minus the reference. Training median: the median of the training patients, "
        "used as a constant estimate.",
        pd.DataFrame(rows),
    )


def table_s5(source, target, results) -> list:
    """S5: overall coverage within VitalDB and on MIMIC-III, and the interval around the training median."""
    n_source = int(source[source.variable == "all"].subjects.iloc[0])
    n_target = int(target[target.variable == "all"].subjects.iloc[0])
    rows = overall_rows(source, f"VitalDB test ({n_source})", ("sbp", "dbp", "mbp"))
    rows += overall_rows(target, f"MIMIC-III ({n_target})", ("sbp",))
    return section(
        "## Table S5. Overall patient-level coverage (%) of the prediction intervals",
        "Intervals were calibrated on the VitalDB calibration patients and, for MIMIC-III, applied unchanged. "
        "Numbers of patients in parentheses.",
        pd.DataFrame(rows),
    ) + median_interval_note(results)


def median_interval_note(results) -> list:
    """The reference interval around the training median, which uses no PPG (explore_review3_analyses.py)."""
    path = results / "review3" / "median_interval.csv"
    if not path.exists():
        return []
    row = pd.read_csv(path).set_index(["variable", "stratum"]).loc[("all", "all")]
    return [
        f"Reference without PPG: a split conformal interval around the training median ({half_up(row['median'], 1)} "
        "mmHg), "
        f"VitalDB test, nominal 90%: coverage {ci(row)}, median width {half_up(row.width_median, 1)} mmHg.",
        "",
    ]


def table_s6(source, target) -> list:
    """S6: coverage by band of true pressure."""
    rows = band_rows(
        source,
        "VitalDB test",
        "lightgbm",
        ("split", "cqr", "mondrian_pred", "mondrian_sqi", "mondrian_pred_sqi", "cqr_mondrian_pred_sqi"),
    )
    rows += band_rows(source, "VitalDB test", "cnn", ("split", "cqr"))
    rows += band_rows(target, "MIMIC-III", "lightgbm", ("split", "cqr"))
    out = [
        "## Table S6. Patient-level coverage (%) of nominal 90% intervals by true pressure (mmHg)",
        "",
        "Values are patient-level coverage with 95% bootstrap confidence intervals. "
        "Each block starts with the number of patients with at least one segment in the range.",
        "",
        "**(a) SBP intervals by true SBP**",
        "",
        markdown(pd.DataFrame(rows)),
        "",
        *dagger_note(rows),
    ]
    rows = band_rows(source, "VitalDB test", "lightgbm", ("split", "cqr"), "dbp", DBP_BANDS, DBP_BAND_LABELS)
    return [*out, "**(b) DBP intervals by true DBP**", "", markdown(pd.DataFrame(rows)), "", *dagger_note(rows)]


def table_s7(source, results, processed) -> list:
    """S7: coverage in subgroups (the values of Fig. 5), and by PPG skewness (planned sensitivity analysis)."""
    parts = []
    for variable, (label, order) in SUBGROUPS.items():
        table = strata_table(source, variable, order)
        table.insert(0, "Variable", label)
        parts.append(table)
    skew = pd.read_csv(results / "conformal" / "skewness_strata.csv")  # evaluate_skewness_strata.py
    table = strata_table(
        skew.assign(target="sbp", level=0.9, model="lightgbm", variable="skewness"),
        "skewness",
        ["low", "middle", "high"],
    )
    table.insert(0, "Variable", "PPG skewness tertile (planned sensitivity analysis)")
    parts.append(table)
    # test patients whose operations fall in different BMI groups appear in each of them
    cases = pd.read_parquet(results / "conformal" / "predictions_test.parquet", columns=["caseid", "subjectid"])
    bmi = pd.read_csv(processed / "vitaldb" / "split.csv", usecols=["caseid", "bmi"])
    cases = cases.drop_duplicates().merge(bmi, on="caseid", how="left")
    cases["group"] = label_bands(cases["bmi"], STRATA["bmi"])
    groups = cases.dropna(subset="group").drop_duplicates(["subjectid", "group"])
    bmi_twice = groups.loc[groups["subjectid"].duplicated(), "subjectid"].nunique()
    number = {1: "One", 2: "Two", 3: "Three"}.get(bmi_twice, str(bmi_twice))
    twice_note = (
        [f"{number} patients had two operations with body mass index in different groups and appear in both."]
        if bmi_twice
        else []
    )
    return [
        "## Table S7. Patient-level coverage (%) of nominal 90% SBP intervals in subgroups, VitalDB test patients, "
        "feature model",
        "",
        "Above / below: share of a patient's segments with the true value above or below the interval, "
        "averaged over patients.",
        "",
        markdown(pd.concat(parts, ignore_index=True)),
        "",
        "† Fewer than 30 patients; reported for reference only.",
        *twice_note,
        "",
    ]


def table_s8(results) -> list:
    """S8: coverage by band of true and of predicted SBP, with the shrinkage model (explore_review_analyses.py)."""
    shrink = pd.read_csv(results / "review" / "shrinkage.csv", index_col=0).reindex(BANDS)
    rows = [
        {
            "SBP band (mmHg)": BAND_LABELS[band],
            "True SBP: patients": int(r.patients_true),
            "True SBP: observed, split": pct(r.observed_by_true),
            "True SBP: shrinkage model, split": pct(r.expected_by_true),
            "Predicted SBP: patients": int(r.patients_predicted),
            "Predicted SBP: split (95% CI)": f"{pct(r.observed_by_predicted)} "
            f"({pct(r.observed_by_predicted_lo)}–{pct(r.observed_by_predicted_hi)})",
            "Predicted SBP: group-wise (95% CI)": f"{pct(r.groupwise_by_predicted)} "
            f"({pct(r.groupwise_by_predicted_lo)}–{pct(r.groupwise_by_predicted_hi)})",
        }
        for band, r in shrink.iterrows()
    ]
    return section(
        "## Table S8. Patient-level coverage (%) of nominal 90% SBP intervals by true and by predicted SBP, "
        "VitalDB test patients, feature model",
        "Shrinkage model: coverage expected if the estimate equals a + b × true SBP plus a normal error "
        "(Section S1.10). Group-wise: calibrated within bands of predicted SBP.",
        pd.DataFrame(rows),
    )


def table_s10(results) -> list:
    """S10: recalibration with MIMIC-III patients, with and without the offset correction."""
    recal = pd.read_csv(results / "external" / "mimic3wdb" / "recalibration.csv")
    recal = recal[recal.model == "lightgbm"].assign(shifted=False)
    shift = pd.read_csv(results / "review" / "recalibration_shift.csv")
    both = pd.concat([recal, shift[shift.shifted]], ignore_index=True)
    rows = []
    for (method, shifted, size), g in both.groupby(["method", "shifted", "subjects"]):
        width = g.width_median.mean()
        unlimited = not np.isfinite(width)  # with 5 patients the interval is unlimited and coverage is meaningless
        coverage = f"{pct(g.coverage.mean())} ({pct(g.coverage.quantile(0.05))}–{pct(g.coverage.quantile(0.95))})"
        rows.append(
            {
                "Method": METHOD_NAMES[method],
                "Offset corrected": "yes" if shifted else "no",
                "Calibration patients": size,
                "Patient-level coverage, mean (5th–95th percentile)": "—" if unlimited else coverage,
                "Draws below 90% (%)": "—" if unlimited else pct((g.coverage < MAIN_LEVEL).mean(), 0),
                "Segments covered at SBP ≥ 160 mmHg (%)": "—" if unlimited else pct(g.coverage_sbp_ge160.mean()),
                "Patient-level coverage at SBP ≥ 160 mmHg (%)": "—"
                if unlimited or "coverage_sbp_ge160_patient" not in g
                else pct(g.coverage_sbp_ge160_patient.mean()),
                "Median width (mmHg)": "unlimited" if unlimited else half_up(width, 1),
            }
        )
    return section(
        "## Table S10. Recalibration with MIMIC-III patients, feature model, nominal 90% SBP intervals "
        "(200 random draws)",
        "Offset corrected: estimates and quantiles first shifted by their mean error in the calibration patients "
        "(exploratory). At SBP ≥ 160 mmHg, two measures are shown: the share of such segments covered, pooled over "
        "patients, and the patient-level coverage (the mean over patients of each patient's share). "
        "With 5 patients the interval is unlimited, so coverage is not shown.",
        pd.DataFrame(rows),
    )


def table_s11(results) -> list:
    """S11: withholding SBP estimates with wide CQR intervals."""
    selective = pd.read_csv(results / "conformal" / "selective.csv")
    part = selective[(selective.target == "sbp") & (selective.model == "lightgbm")]
    rows = []
    for _, r in part.iterrows():
        tiny = r.accepted < MIN_KEPT_SHARE
        rows.append(
            {
                "Dataset": DATASET_NAMES[r.dataset],
                "Width threshold (mmHg)": half_up(r.width_threshold, 1),
                "Segments kept (%)": pct(r.accepted),
                "MAE (mmHg)": "—" if tiny else half_up(r.mae, 1),
                "SD of error (mmHg)": "—" if tiny else half_up(r.sd, 1),
                "Coverage of kept (%)": "—" if tiny else pct(r.coverage_subject),
                "SD of true SBP (mmHg)": "—" if tiny else half_up(r.true_sd, 1),
            }
        )
    return section(
        "## Table S11. Withholding SBP estimates with wide CQR intervals, feature model",
        "Thresholds were chosen on the VitalDB calibration patients to keep 10%, 20%, …, 100% of their segments. "
        "Values are not shown when fewer than 1% of segments were kept.",
        pd.DataFrame(rows),
    )


def table_s12a(results) -> list:
    """S12(a): paired differences between methods (explore_review_analyses.py)."""
    diff = pd.read_csv(results / "review" / "differences.csv")
    rows = [
        {
            "Dataset": r.dataset,
            "Subgroup": r.stratum,
            "Comparison": COMPARISON_NAMES.get(r.comparison, r.comparison),
            "Patients": int(r.patients),
            "Difference (percentage points, 95% CI)": f"{pct(r.difference)} ({pct(r.ci_low)} to {pct(r.ci_high)})",
        }
        for r in diff.itertuples()
    ]
    return [
        "## Table S12. Differences between methods and models, nominal 90% SBP intervals",
        "",
        "Patient-level coverage (%). Confidence intervals from 2,000 bootstrap resamples of patients; "
        "in (a), each patient's pair of values is kept together.",
        "",
        *section("**(a) Paired differences between methods, feature model**", None, pd.DataFrame(rows)),
    ]


def table_s12b(results) -> list:
    """S12(b): exploratory alternatives (group-wise by PPG pulse rate, shrinkage correction)."""
    rows = []
    for name, path, method in (
        ("Split conformal", "pulse_groupwise.csv", "split"),
        ("Group-wise (PPG pulse rate)", "pulse_groupwise.csv", "group-wise (PPG pulse rate)"),
        ("Shrinkage corrected, split", "deshrink.csv", None),
    ):
        frame = pd.read_csv(results / "review2" / path)
        if method:
            frame = frame[frame.method == method]
        for dataset in ("VitalDB test", "MIMIC-III"):
            part = frame[frame.dataset == dataset].set_index(["variable", "stratum"])
            row = {"Method": name, "Dataset": dataset, "All (95% CI)": ci(part.loc[("all", "all")])}
            for band in ("<90", "≥160"):
                row[f"True SBP {BAND_LABELS[band]} (95% CI)"] = ci(part.loc[("true_sbp", band)])
            row["Median width (mmHg)"] = half_up(part.loc[("all", "all")].width_median, 1)
            rows.append(row)
    return section(
        "**(b) Exploratory alternatives, feature model (not planned in advance)**",
        "Group-wise (PPG pulse rate): calibrated within bands of the pulse rate measured from the PPG "
        "(<60, 60–79, 80–99, ≥100 per minute). Shrinkage corrected: the estimate was mapped back by (estimate − a)/b "
        "with a and b from the calibration patients (Section S1.10), and a split conformal interval was built around "
        "it. Width is the median over segments; for group-wise calibration it is that of all segments.",
        pd.DataFrame(rows),
    )


def signed_text(text: str) -> str:
    """Replace the hyphen of a negative number by a true minus sign."""
    return text.replace("-", "−")


def table_s12c(results) -> list:
    """S12(c): waveform model minus feature model on the same patients (explore_model_comparison.py)."""
    comp = pd.read_csv(results / "review2" / "model_comparison.csv")
    rows = []
    for r in comp.itertuples():
        if r.measure.startswith("MAE"):
            value, difference = (lambda v: half_up(v, 1)), (lambda v: signed_text(half_up(v, 2)))
        else:
            value, difference = pct, (lambda v: signed_text(pct(v)))
        rows.append(
            {
                "Dataset": r.dataset,
                "Subgroup": COMPARISON_STRATA[r.stratum],
                "Measure": MEASURE_NAMES[r.measure],
                "Patients": int(r.patients),
                "Feature model": value(r.feature),
                "Waveform model": value(r.waveform),
                "Difference (95% CI)": f"{difference(r.difference)} ({difference(r.ci_low)} to "
                f"{difference(r.ci_high)})",
            }
        )
    return section(
        "**(c) Waveform model − feature model (exploratory, not planned in advance)**",
        "The same patients for both models. MAE is over segments, with patients resampled; coverage is per patient, "
        "with each patient's pair of values kept together. Differences in coverage are in percentage points.",
        pd.DataFrame(rows),
    )


WEIGHTING_NAMES = {
    "all features (main analysis)": "All 42 features (main analysis)",
    "without b/a and upslope time": "Without b/a and upslope time",
    "rhythm features only": "Rhythm features only",
    "all features, ratio clipped to 0.1-10": "All features, ratio clipped to 0.1–10",
}


def mean_range(values) -> str:
    """Mean and 2.5th to 97.5th percentile, in %."""
    return f"{pct(values.mean())} ({pct(values.quantile(0.025))}–{pct(values.quantile(0.975))})"


def _s13_overshoot(folder) -> list:
    """S13(a): segments with a pulse pressure above 100 mmHg, and coverage without them."""
    pp = pd.read_csv(folder / "pulse_pressure.csv")
    share = pp[pp.analysis == "share_above_100"].set_index("stratum")["value"]
    rows = [
        {"Analysis": "Segments with pulse pressure > 100 mmHg (%)", **{BAND_LABELS[b]: pct(share[b]) for b in BANDS}}
    ]
    for method in ("split", "cqr"):
        part = pp[(pp.analysis != "share_above_100") & (pp.method == method) & (pp.variable == "true_sbp")]
        part = part.set_index("stratum")
        rows.append(
            {
                "Analysis": f"{METHOD_NAMES[method]}, without these segments",
                **{BAND_LABELS[b]: ci(part.loc[b]) for b in BANDS},
            }
        )
    return section(
        "**(a) Possible overshoot of the arterial line: segments with a pulse pressure above 100 mmHg, by true SBP "
        "(mmHg), removed from calibration and test**",
        None,
        pd.DataFrame(rows),
    )


def _s13_resplit(folder) -> list:
    """S13(b): coverage over repeated random splits of the calibration and test patients."""
    resplit = pd.read_csv(folder / "resplit.csv")
    rows = []
    for method, part in resplit.groupby("method", sort=False):
        rows.append(
            {
                "Method": {"mondrian_pred": "Group-wise (predicted SBP)"}.get(method, METHOD_NAMES.get(method, method)),
                "All": mean_range(part["all"]),
                "True SBP ≥160": mean_range(part["true ≥160"]),
                "Predicted SBP 140–159": mean_range(part["predicted 140–<160"]),
                "Predicted SBP ≥160": mean_range(part["predicted ≥160"]),
            }
        )
    return section(
        f"**(b) Variation with the calibration patients: the 1,254 calibration and test patients split again at random "
        f"into two halves {resplit.repeat.nunique()} times; mean (2.5th–97.5th percentile) over the splits**",
        None,
        pd.DataFrame(rows),
    )


def _s13_offset(folder) -> list:
    """S13(c): one reference reading per patient."""
    offset = pd.read_csv(folder / "offset_calibration.csv")
    rows = []
    for (n_first, dataset, method), group in offset.groupby(["reference_segments", "dataset", "method"], sort=False):
        part = group.set_index(["variable", "stratum"])
        overall = part.loc[("all", "all")]
        rows.append(
            {
                "Reference": {1: "10 s", 6: "1 min", 30: "5 min"}.get(n_first, str(n_first)),
                "Dataset": dataset,
                "Method": METHOD_NAMES[method],
                "MAE (mmHg)": half_up(overall.mae, 1),
                "All": ci(overall),
                "True SBP <90": ci(part.loc[("true_sbp", "<90")]),
                "True SBP ≥160": ci(part.loc[("true_sbp", "≥160")]),
                "Median width (mmHg)": half_up(overall.width_median, 1),
            }
        )
    return section(
        "**(c) One reference reading per patient: the mean of reference minus estimate over each patient's first "
        "segments (10 s, 1 min or 5 min) added to the patient's later estimates; intervals calibrated on the "
        "corrected calibration patients**",
        None,
        pd.DataFrame(rows),
    )


def _s13_weighting(folder, results) -> list:
    """S13(d): weighting with domain classifiers on fewer features or with tighter clipping."""
    weighting = pd.read_csv(folder / "weighting_subsets.csv")
    plain = pd.read_csv(results / "external" / "mimic3wdb" / "coverage.csv")
    plain = plain[
        (plain.model == "lightgbm") & (plain.target == "sbp") & (plain.level == MAIN_LEVEL) & (plain.variable == "all")
    ].set_index("method")["coverage"]
    rows = [
        {
            "Classifier": WEIGHTING_NAMES[r.variant],
            "AUC": half_up(r.auc, 3),
            "Effective calibration patients (%)": pct(r.effective_patient_share),
            "Method": METHOD_NAMES[r.method.removeprefix("weighted_")],
            "Coverage on MIMIC-III": ci(r),
            "Unlimited intervals (%)": pct(r.infinite_share),
            "Median width (mmHg)": "unlimited" if not np.isfinite(r.width_median) else half_up(r.width_median, 1),
        }
        for r in weighting.itertuples()
    ]
    return section(
        "**(d) Weighting with unlabelled MIMIC-III data: domain classifiers on fewer features or with the density "
        "ratio clipped to 0.1–10 (main analysis: all 42 features, ratio clipped to 0.001–1000)**",
        "Effective calibration patients: Kish effective sample size of the calibration patients, with the weights "
        "summed per patient, as a share of all calibration patients. Without weighting, coverage on MIMIC-III was "
        f"{pct(plain['split'])}% (split conformal) and {pct(plain['cqr'])}% (CQR).",
        pd.DataFrame(rows),
    )


def _s13_seeds(folder) -> list:
    """S13(e): the waveform model trained with other seeds."""
    seeds = pd.read_csv(folder / "cnn_seeds.csv")
    seeds = seeds[seeds.target == "sbp"]
    rows = [
        {
            "Run": "main" if r.run == "waveform_model" else r.run.replace("waveform_model_", ""),
            "MAE (mmHg)": half_up(r.mae, 1),  # the source file has 3 decimals; 1 decimal avoids rounding twice
            "Mean error ± SD (mmHg)": f"{signed_text(half_up(r.me, 1))} ± {half_up(r.sd, 1)}",
        }
        for r in seeds.itertuples()
    ]
    return section(
        "**(e) Waveform model trained with five other seeds: SBP accuracy in the 376 validation patients**",
        None,
        pd.DataFrame(rows),
    )


def table_s13(results) -> list:
    """S13: analyses added after the third review (explore_review3_analyses.py). All exploratory."""
    folder = results / "review3"
    return [
        "## Table S13. Additional exploratory analyses, nominal 90% SBP intervals, feature model unless stated",
        "",
        "Patient-level coverage (%) with 95% bootstrap confidence intervals unless stated. None of these analyses was "
        "planned in advance.",
        "",
        *_s13_overshoot(folder),
        *_s13_resplit(folder),
        *_s13_offset(folder),
        *_s13_weighting(folder, results),
        *_s13_seeds(folder),
    ]


def table_s14(results) -> list:
    """S14: MAE with 95% confidence intervals from a bootstrap over patients (error bars of Fig. 3(b))."""
    acc = pd.read_csv(results / "review" / "test_accuracy.csv")
    rows = [
        {
            "Target": r.target.upper(),
            "Model": r.predictor,
            "MAE (95% CI)": f"{half_up(r.mae, 1)} ({half_up(r.mae_lo, 1)}–{half_up(r.mae_hi, 1)})",
        }
        for r in acc.itertuples()
    ]
    return section(
        "## Table S14. Mean absolute error with 95% confidence intervals, VitalDB test patients (mmHg)",
        "Confidence intervals from 2,000 bootstrap resamples of patients.",
        pd.DataFrame(rows),
    )


def table_s15(results) -> list:
    """S15: intervals sorted against 140 mmHg (true SBP >= 140) and 65 mmHg (true MBP < 65) (fig. S10)."""
    ct = pd.read_csv(results / "conformal" / "exploratory_clinical_threshold.csv").set_index(
        ["dataset", "model", "method", "subset"]
    )
    hy = pd.read_csv(results / "review4" / "hypotension.csv").set_index(["dataset", "model", "method"])
    rows = []
    for name in ("VitalDB test", "MIMIC-III"):
        for model in ("lightgbm", "cnn"):
            for method in ("split", "cqr"):
                r = ct.loc[(name, model, method, "true_sbp_>=140")]
                h = hy.loc[(name, model, method)]
                rows.append(
                    {
                        "Dataset": name,
                        "Model": MODEL_NAMES[model],
                        "Method": METHOD_NAMES[method],
                        "SBP ≥ 140: wholly ≥ 140 (%)": pct(r.above),
                        "SBP ≥ 140: contains 140 (%)": pct(r.undecided),
                        "SBP ≥ 140: wholly < 140 (%)": pct(r.below),
                        "MBP < 65: wholly < 65 (%)": pct(h.low_wholly_below),
                        "MBP < 65: contains 65 (%)": pct(h.low_undecided),
                        "MBP < 65: wholly ≥ 65 (%)": pct(h.low_wholly_above),
                    }
                )
    return section(
        "## Table S15. Nominal 90% intervals sorted against 140 mmHg (true SBP ≥ 140 mmHg) and 65 mmHg "
        "(true MBP < 65 mmHg)",
        "Shares of segments, averaged first within each patient. Exploratory.",
        pd.DataFrame(rows),
    )


def table_s16(results) -> list:
    """S16: distribution of the true SBP and of the coverage of each patient (fig. S9), as aggregate counts only."""
    sbp = {
        "VitalDB test": pd.read_parquet(results / "conformal" / "predictions_test.parquet", columns=["sbp"])["sbp"],
        "MIMIC-III": pd.read_parquet(results / "external" / "mimic3wdb" / "predictions.parquet", columns=["sbp"])[
            "sbp"
        ],
    }
    edges = SBP_BIN_EDGES  # shared with plot_supplement.py
    for name, values in sbp.items():  # every segment must fall in a bin, so that the shares sum to 100%
        if values.min() < edges[0] or values.max() > edges[-1]:
            raise ValueError(f"{name}: true SBP outside {edges[0]}–{edges[-1]} mmHg")
    shares = {name: np.histogram(values, bins=edges)[0] / len(values) for name, values in sbp.items()}
    rows = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        row = {"True SBP (mmHg)": f"{lo}–<{hi}" if k < len(edges) - 2 else f"{lo}–{hi}"}
        for name, share in shares.items():
            row[f"{name} (% of segments)"] = pct(share[k], 2)
        rows.append(row)
    patients = pd.read_csv(results / "review4" / "patient_coverage.csv")
    edges_c = COVERAGE_BIN_EDGES
    rows_c = []
    for k in range(len(edges_c) - 1):
        lo, hi = edges_c[k], edges_c[k + 1]
        row = {"Coverage of the patient (%)": f"{lo:.0f}–<{hi:.0f}" if k < len(edges_c) - 2 else f"{lo:.0f}–100"}
        for (dataset, method), part in patients.groupby(["dataset", "method"], sort=False):
            counts, _ = np.histogram(100 * part.coverage, bins=edges_c)
            row[f"{dataset}, {METHOD_NAMES[method]} (% of patients)"] = pct(counts[k] / len(part), 1)
        rows_c.append(row)
    return [
        "## Table S16. Distribution of the true SBP and of the coverage of each patient",
        "",
        "Nominal 90% SBP intervals, feature model. Aggregate shares only.",
        "",
        "**(a) True SBP, share of segments per 5 mmHg**",
        "",
        markdown(pd.DataFrame(rows)),
        "",
        "**(b) Coverage of each patient, share of patients per 5 percentage points**",
        "",
        markdown(pd.DataFrame(rows_c)),
        "",
    ]


def main():
    config = load_config()
    results = PROJECT_ROOT / config["paths"]["results"]
    processed = PROJECT_ROOT / config["paths"]["processed"]
    source = pd.read_csv(results / "conformal" / "coverage.csv")
    target = pd.read_csv(results / "external" / "mimic3wdb" / "coverage.csv")
    # Tables appear in the order in which the main text first cites them (S6 before S7).
    out = [
        "# Supplementary tables (draft)",
        "",
        "<!-- Written by scripts/make_tables.py from the result files; do not edit by hand. -->",
        "",
        *table_s1(processed),
        *tables_s2_s3(),
        *table_s4(results),
        *table_s5(source, target, results),
        *table_s6(source, target),
        *table_s7(source, results, processed),
        *table_s8(results),
        *table_s10(results),
        *table_s11(results),
        *table_s12a(results),
        *table_s12b(results),
        *table_s12c(results),
        *table_s13(results),
        *table_s14(results),
        *table_s15(results),
        *table_s16(results),
    ]
    out_path = results / "tables" / "supplement_tables.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out), encoding="utf-8")
    print("Wrote the supplementary tables (headings follow an earlier numbering; README.md maps them to the paper)")


if __name__ == "__main__":
    main()
