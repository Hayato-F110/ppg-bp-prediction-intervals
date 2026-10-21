# Analysis plan

> **About this translation.** This is an English translation of the analysis plan, which was written in Japanese
> (`docs/analysis_plan.md` of the author's working repository). The Japanese text is the original. It was approved by
> the author on 13 October 2026, before any model was fitted. The translation was made on 21 October 2026 with the
> help of a generative AI assistant and checked against the original. It keeps the section numbers, so that the code
> can refer to them ("analysis plan, Section 4").
>
> History of the original text (from the version-control history):
> - 13 October 2026: first version, approved. Section 4.4 added and then aligned with Orphanidou et al. (2015) on the
>   same evening, before any model was fitted; both approved (see "Change history" at the end).
> - 14 October 2026, 07:19 (JST): the reference of the jSQI criteria in Section 2.1 was corrected to Sun, Reisner and
>   Mark (2006), and a note was added that the original paper had been read. No criterion changed. This edit is not
>   in the change history of the original.
> - No later change.
>
> Notes in square brackets are the translator's. "Phase 2" to "Phase 5" are the stages of the project: data
> preparation, baseline models, conformal prediction within a dataset, and evaluation across datasets.

- Written: 13 October 2026. **Approved by the author on the same day.** The approved version can be found in the
  version-control history.
- Purpose: to fix the evaluation, the subgroup limits and the main outcome before seeing results. Any later change is
  recorded with its date and reason in the "Change history", and is made only after the author has approved it.
- What had been seen when this plan was written: the distributions of the reference blood pressure and of the patient
  characteristics (summaries over training, calibration and test together). **No model estimate or interval had been
  made yet.**

## 1. Question and hypotheses

**Main question:** When conformal prediction intervals for PPG-only cuffless blood-pressure estimation are applied to
another dataset, where, by how much and why do they lose coverage, and how much data from the target dataset restores
it?

| Label | Hypothesis | Quantity used to judge it |
|---|---|---|
| H1 (confirmatory) | Within a dataset, the nominal coverage is reached; across datasets, coverage falls below it | Patient-level coverage and its 95% confidence interval |
| H2 (central) | The loss of coverage is large in the hypertensive and hypotensive ranges, and the misses are one-sided (at high pressures, the true value lies above the interval) | Coverage by band of true SBP; rates of misses above and below |
| H3 | Intervals stratified by estimated blood-pressure band or by signal quality (Mondrian type) correct the bias of coverage by true blood-pressure band only partly | Difference, with and without stratification, in coverage by band of true SBP |
| H4 | Recalibration on a few patients of the target dataset restores the nominal coverage without retraining the model. Weighting that uses only unlabelled target data does not restore it fully | Coverage by number of patients used for calibration |

Results that do not agree with the hypotheses are reported as they are.

## 2. Data

- Source dataset: VitalDB (`data/processed/vitaldb/`). 10-s segments, PPG at 125 Hz. The reference values are the means
  of the beat-by-beat SBP and DBP within the segment, and the time average of the arterial pressure (MBP).
- Kept segments: 3,653,817; patients: 3,134.
- **The split is fixed**: training 1,880, calibration 627, test 627 patients (by patient, seed 20261013). The SHA-256 of
  `split.csv` is recorded in the work log. It is not to be made again. [The split is published as
  `data_lists/vitaldb_split.csv`.]
- External data: not yet decided (whether to use data derived from MIMIC is decided by the author before Phase 5).
  As a rule, all segments of the external data are used for testing. Only the recalibration experiment (Section 6.3)
  uses some of them for calibration.

### 2.1 Exclusion criteria for segments (already applied)

The criteria are checked from the top; a segment is excluded for the first reason that applies.

| Reason | Criterion | Basis |
|---|---|---|
| Missing values | Missing values in the PPG or the arterial pressure | — |
| Arterial pressure out of range | Minimum within the segment below 20 mmHg, or maximum above 300 mmHg | The same as the range of "non-physiological pressure" of jSQI (Sun, Reisner, Mark 2006, Computers in Cardiology 33:13–16; implemented in the PhysioNet Cardiovascular Signal Toolbox) |
| Flat line | The same value for 1 s or longer | The same as the criterion of PulseDB (Wang et al. 2023) |
| Too few beats | Fewer than 5 beats detected in the arterial pressure | Set in this study |
| Pulse rate out of range | Below 40 or above 180 per minute | jSQI uses 20–200; this study sets a narrower range. Rule 1 of Orphanidou et al. 2015 (DOI 10.1109/JBHI.2014.2338351) gives the same 40–180 per minute as "the range possible in adults with wearable sensors" (added on 13 October 2026 after reading the original paper; the criterion itself did not change) |
| Variability of beat-by-beat SBP | Standard deviation of beat-by-beat SBP within the segment above 15 mmHg | jSQI treats a beat-to-beat change above 20 mmHg as abnormal. This study replaces it with a criterion per segment |
| Small pulse pressure | SBP − DBP below 20 mmHg | The same as the value (20 mmHg) in the code of jSQI |

- The values of jSQI were checked by reading the code published on PhysioNet. **The original paper (Sun, Reisner,
  Mark, Computers in Cardiology 2006;33:13–16) was also read on 14 October 2026.** The criteria of its Table 2 include
  systolic above 300 mmHg, diastolic below 20 mmHg, mean below 30 or above 200, heart rate below 20 or above 200, pulse
  pressure below 20 mmHg, and beat-to-beat changes of systolic or diastolic pressure above 20 mmHg. The "arterial
  pressure out of range" and "small pulse pressure" criteria of this study agree with the values of that paper.
- Unlike PulseDB, **segments are not excluded by the correlation between the PPG and the arterial pressure, nor by the
  quality of the PPG**. PPG quality is treated as a subgroup of the conditional evaluation, not as an exclusion
  (Section 4.2 [the subgroups are listed in Section 4.3]). In real use there is no arterial pressure, so the PPG cannot
  be screened with it.
- Sensitivity analysis: check whether the main results change when the pulse-rate range is widened to 20–200, and when
  the variability limit is set to 10 or 20 mmHg.

## 3. Models

Two kinds of model are handled with the same split and the same evaluation. Which one is the main model is decided
after their accuracy has been measured without the test data (by validation within the training data).

- Waveform model: a small-to-medium one-dimensional convolutional network. Input: 10 s of PPG.
- Feature model: LightGBM. Inputs: morphological features of the PPG, fiducial points of the acceleration
  plethysmogram (amplitude ratios and times of the a to e waves), pulse intervals, and signal-quality indices. Linear
  regression for comparison.
- Versions with and without age and sex are both evaluated. The main analysis uses the version without them (PPG
  only).
- For training, at most 400 segments per patient are drawn at random (the same limit as PulseDB; this keeps training
  from leaning towards patients with many segments).
- Model selection and tuning use only validation data, made by splitting the training patients further by patient.
  **The test data are not used at all until the final evaluation of Phase 4.**

## 4. Prediction intervals

### 4.1 Methods

| Method | Description |
|---|---|
| Split conformal | Intervals of fixed width from a quantile of the absolute residuals |
| CQR | Conformal correction of the outputs of quantile regression (the width varies with the input) |
| Mondrian type | Separate correction in each stratum. Strata: band of estimated SBP, and signal quality (both known from the input at test time) |

- Nominal coverage: 80%, **90% (main)**, 95%.
- Calibration uses the calibration patients (627). **Each patient has equal weight** (a weighted quantile, with a
  weight of 1 / the number of segments of that patient on each segment, so that patients with many segments do not
  dominate the correction). Treating all segments with equal weight is a sensitivity analysis.

### 4.2 Measures

**Main outcome: SBP, nominal 90%, patient-level coverage.** Coverage is computed for each patient and averaged over
patients. Its 95% confidence interval comes from a bootstrap over patients (2,000 resamples).

Also reported:

1. Segment-level coverage (all segments pooled).
2. **The rates of misses above and below** (the share of true values above the interval, and below it).
3. Median and interquartile range of the interval width.
4. Distribution of coverage across patients. For a nominal 90%, the share of patients with coverage below 80% and
   below 70%.
5. The same measures for DBP and MBP (secondary).

Coverage and interval width are always shown together. Wider intervals raise coverage, so neither one alone is an
evaluation.

### 4.3 Subgroups of the conditional evaluation (limits fixed)

| Variable | Strata |
|---|---|
| True SBP (mmHg) | below 90 / 90–99 / 100–119 / 120–139 / 140–159 / 160 or above |
| True DBP (mmHg) | below 50 / 50–59 / 60–69 / 70–79 / 80–89 / 90 or above |
| Age (years) | below 40 / 40–59 / 60–74 / 75 or above |
| Sex | male / female |
| BMI | below 18.5 / 18.5–24.9 / 25–29.9 / 30 or above |
| Pulse rate (per minute) | below 60 / 60–79 / 80–99 / 100 or above |
| Signal quality | Tertiles in the training data. The index is the agreement of pulse shapes (Section 4.4 below; defined on 13 October 2026) |
| Position within the patient | Difference from the patient's median SBP: below −20 / within ±20 / above +20 |

- For reference, the shares of segments by SBP band in all of VitalDB: below 90, 6.4%; 90–99, 12.5%; 100–119, 39.5%;
  120–139, 28.8%; 140–159, 10.0%; 160 or above, 2.8%.
- The last subgroup, "position within the patient", shows whether the intervals follow changes of blood pressure
  within a person. In VitalDB, the standard deviation of SBP within patients (median 15.7 mmHg) is larger than the
  standard deviation of the patient means (11.9 mmHg), so there is enough change within patients.
- The confidence interval of each stratum also comes from a bootstrap over patients. A stratum with fewer than 30
  patients is treated as a reference value and is not used for conclusions.

### 4.4 Definition of the signal-quality index (added on 13 October 2026, before the features or estimates of the test data were seen)

- **Main index: agreement of pulse shapes (`sqi_template`), following the template-matching procedure of Orphanidou
  et al. 2015 (IEEE JBHI, DOI 10.1109/JBHI.2014.2338351).** Within a 10-s segment: (1) find the median interval
  between the detected peaks; (2) cut a window of that width centred on each peak; (3) take the mean of all pulses as
  the template; (4) compute the correlation coefficient of each pulse with the template, and use their mean as the
  index. The closer to 1, the more alike the pulse shapes. Implemented as `template_sqi` in `src/features/extract.py`
  [now `src/ppg_bp/features/extract.py`].
- Differences from that paper: (a) the paper sets a threshold on the index to decide "usable / unusable"; this study
  uses no threshold and divides the continuous values into tertiles. (b) The rules on the plausibility of heart rate
  and beat intervals that the paper also uses are not applied (the pulse-rate range is already checked by the
  exclusion based on arterial pressure). (c) A segment with fewer than 3 pulses whose window fits in the segment gets 0
  (added in this study; segments whose quality cannot be assessed go to the lowest-quality stratum).
- The idea of assessing PPG quality by the correlation of each pulse with a template also appears in Li and Clifford
  2012 (Physiol. Meas. 33:1491, DOI 10.1088/0967-3334/33/9/1491).
- The limits of the strata are the tertiles computed **on the segments of the training patients only**. The same
  limits are used for the calibration, test and external data.
- Reasons for this index: (1) it can be computed from the PPG alone at test time. (2) It does not depend on the pulse
  shape itself, only on how alike the pulses are within a segment, so its meaning changes little even when the
  waveform shape differs between datasets (checked in step 1 of Phase 3).
- Sensitivity analysis: results stratified by skewness (`sqi_skewness`; the best of 8 indices in Elgendi 2016,
  Bioengineering 3(4):21, and also used for exclusion by PulseDB) are also shown. Skewness depends on the waveform
  shape, so it is not the main index.
- Status of the sources (13 October 2026): **Orphanidou et al. 2015 was checked in the original PDF** (obtained by the
  author). The implementation agrees with steps 1) to 4) of the original (median beat interval from all detected
  peaks / a window of that width centred on each peak / the mean of all pulses as the template / the correlation of
  each pulse with the template, averaged over the segment). That paper also uses 10-s segments. Its threshold for PPG
  is 0.86 (not used in this study). The paper applies three plausibility rules before the correlation: heart rate
  40–180 per minute, at most 3 s between peaks, and a ratio below 2.2 between the longest and shortest beat intervals.
  This study does not build these into the index (the first is the same range as the exclusion criterion based on
  arterial pressure). The secondary sources used before that were two papers in Sensors 2023 (DOI 10.3390/s23042220,
  10.3390/s23052561). Li and Clifford 2012: abstract only, full text not read. Elgendi 2016: abstract checked by
  search.

## 5. Number of patients needed for calibration (within the dataset)

The number of calibration patients is set to 10, 20, 45, 100, 200, 400 and 627. For each, the patients are drawn
again at random from the calibration patients 200 times, and the distribution of patient-level coverage on the test
data is obtained. 45 patients is the number used for calibration in the closest previous study (Moulaeifard et al.
2026).

## 6. Evaluation across datasets

### 6.1 Applied as they are

The model and intervals trained and calibrated on VitalDB are applied to the external data without change, and the
measures of Sections 4.2 and 4.3 are computed.

### 6.2 Corrections that use unlabelled target data

- Weighted conformal prediction (correction of the shift in the input distribution). The density ratio is estimated by
  a classifier that tells the source from the target.
- A correction that uses the distribution of the estimated blood pressure in the target data (for the shift in the
  blood-pressure distribution). Secondary.

### 6.3 Recalibration on a few target patients

5, 10, 20, 50 and 100 patients are drawn at random from the external data and used for calibration; the rest are used
for testing. Each is repeated 200 times. The model is not retrained.

### 6.4 Separating the causes of the shift

- Shift in the input: accuracy (AUC) of a classifier that tells the source from the target.
- Shift in the blood-pressure distribution: a distance between the distributions.
- The external data are reweighted so that their blood-pressure distribution matches that of the source, and coverage
  is computed again. The loss that remains is the part due to factors other than the blood-pressure distribution.

## 7. Withholding (selective prediction)

Segments whose interval is wider than a threshold get no estimate. As the threshold varies, curves of the following
against the share of segments accepted are shown: mean absolute error; mean error and standard deviation of the error
(with reference lines at 5 mmHg and 8 mmHg); and coverage in the accepted segments. The threshold is set on the
calibration data only. The share accepted is always shown alongside.

## 8. Statistical approach

- The statistical unit is the patient. Segments are not independent, so no test or confidence interval that treats
  segments as independent is used.
- There are many strata, so differences in single strata are not claimed by significance. They are described by the
  size of the effect and its confidence interval.
- Random seeds are fixed in the configuration files.

## 9. Limitations accepted from the start

- The data come from patients during surgery under general anaesthesia and do not represent daily life.
- The PPG is from the finger clip of a hospital monitor and is processed inside the device, so it may not represent
  the waveforms of wearable devices (Charlton et al. 2023, Section 21).
- The reference is intra-arterial pressure, which differs systematically from a cuff.
- There is no information on skin colour, and the population is almost a single ethnic group.
- Contact pressure, hand height and temperature were not recorded.

## Change history

| Date | Change | Reason | Approval by the author |
|---|---|---|---|
| 13 Oct 2026 | First version | — | **Approved (13 Oct 2026).** This version is the baseline |
| 13 Oct 2026, evening | Section 4.4 added (definition of the signal-quality index); the corresponding row of the table in Section 4.3 updated | The first version had planned this item: "decided in Phase 3, before seeing the test data, and added". Not a change of the evaluation | **Approved (13 Oct 2026, evening).** Decided under the author's instruction to keep going, then submitted for approval afterwards and approved ("yes"). This was before any model estimate was made |
| 13 Oct 2026, evening | The procedure of the main index in Section 4.4 aligned with that of Orphanidou et al. 2015 (window changed from "each pulse from start to end, resampled to 50 points" to "a window of the width of the median peak interval"; template changed from the median to the mean) | The author's condition: "OK if it follows previous work". The first definition had the same idea but its own procedure. No analysis had yet used the values of the index | **Approved (13 Oct 2026, evening).** The result of the comparison with the original paper was reported, and the author answered "yes" |
