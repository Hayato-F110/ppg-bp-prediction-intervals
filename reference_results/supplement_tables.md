# Supplementary tables (draft)

<!-- Written by scripts/make_tables.py from the result files; do not edit by hand. -->

## Table S1. Segments kept and excluded

Rules were applied in the order shown; a segment is counted under the first rule that it failed. In VitalDB, 206 of 3,340 patients had no segment left.

| Outcome | VitalDB segments | VitalDB % | MIMIC-III segments | MIMIC-III % |
|---|---|---|---|---|
| Missing samples | 173,268 | 3.8 | 3,241 | 4.0 |
| Arterial pressure out of range | 512,002 | 11.2 | 9,264 | 11.4 |
| Constant signal for 1 s or longer | 38 | 0.0 | 6,428 | 7.9 |
| Fewer than five beats | 53,228 | 1.2 | 1,462 | 1.8 |
| Pulse rate out of range | 11,013 | 0.2 | 338 | 0.4 |
| Unstable systolic pressure | 160,051 | 3.5 | 5,733 | 7.1 |
| Pulse pressure below 20 mmHg | 12,030 | 0.3 | 854 | 1.1 |
| Kept | 3,653,817 | 79.9 | 53,928 | 66.4 |

## Table S2. The 42 features of the feature model

Beat-level values are summarised by their median over the beats of a 10-s segment. Rhythm and signal-quality features describe the whole segment.

| Feature | Group | Definition |
|---|---|---|
| n_beats | Signal quality | Number of detected beats in the segment |
| b_a, c_a, d_a, e_a, f_a | APG | Amplitude of each APG wave divided by that of the a wave |
| aging_index | APG | (b − c − d − e)/a |
| apg_index | APG | (c + d − b)/a |
| b_a_iqr | APG | Interquartile range of b/a over the beats (at least four beats) |
| t_ab, t_ac, t_ad, t_ae, t_af | APG | Time from the a wave to each later wave (s) |
| t_be, t_de | APG | Time from the b and d waves to the e wave (s) |
| t_ae_rel | APG | Time from a to e divided by the beat duration |
| crest_time, crest_time_rel | Pulse shape | Time from onset to systolic peak (s), and the same divided by the beat duration |
| upslope_time, max_slope | Pulse shape | Time from onset to the steepest upslope (s), and that slope divided by the pulse amplitude |
| sw25, sw50, sw75, dw25, dw50, dw75 | Pulse shape | Width before (sw) and after (dw) the peak at 25%, 50% and 75% of the pulse amplitude (s) |
| width50_rel, dw_sw50 | Pulse shape | Relative pulse width (width at 50% divided by the beat duration), and the width after the peak divided by that before it |
| area_ratio | Pulse shape | Area after the peak divided by the area before it |
| t_peak_e, height_e | Pulse shape | Time from the systolic peak to the e wave (s), and the pulse height there relative to the peak |
| ibi_median, ibi_sd, ibi_rmssd, ibi_cv | Rhythm | Median, SD, root mean square of successive differences, and coefficient of variation of the pulse interval |
| sqi_template | Signal quality | Mean correlation of each pulse with the average pulse of the segment |
| sqi_skewness, sqi_kurtosis | Signal quality | Skewness and kurtosis of the filtered PPG |
| frac_e_found, frac_cd_found, frac_cd_distinct | Signal quality | Share of beats in which the e wave, the c and d waves, and distinct c and d waves were found |

## Table S3. Model settings

| Model | Item | Setting |
|---|---|---|
| Both | Training data | At most 400 random segments per training patient; 20% of the training patients held out for model selection |
| Both | Outputs | Mean, and quantiles 0.025, 0.05, 0.1, 0.9, 0.95, 0.975 |
| Feature model | Algorithm | LightGBM gradient-boosted trees; one model for the mean (squared error) and one per quantile (quantile loss) |
| Feature model | Settings | Learning rate 0.05; up to 3000 trees, early stopping after 100 rounds; 63 leaves; at least 200 segments per leaf; row and feature subsampling 0.8 and 0.8; L2 penalty 1.0 |
| Feature model | Missing features | Handled by LightGBM without imputation |
| Waveform model | Input | Raw 10-s PPG at 125 Hz, standardised per segment |
| Waveform model | Architecture | 1-D convolution (15 taps, stride 2); four stages of two residual blocks (kernel 7) with 32, 48, 64, 96 channels; global average pooling; linear output; about 450,000 parameters |
| Waveform model | Loss | Squared error for the mean plus pinball loss for the quantiles, on standardised targets |
| Waveform model | Training | AdamW, learning rate 0.002 with a one-cycle schedule, weight decay 0.0001, batch 256, 150,000 random segments per epoch, at most 30 epochs, early stopping after 5 epochs without improvement |
| Both | Crossing quantiles | Lower and upper quantiles swapped when they crossed |

## Table S4. Accuracy of the point estimate, 627 VitalDB test patients (724,202 segments), PPG-only inputs (mmHg)

Error is the estimate minus the reference. Training median: the median of the training patients, used as a constant estimate.

| Target | Model | MAE | Mean error ± SD | Within 5 / 10 / 15 mmHg (%) |
|---|---|---|---|---|
| SBP | Training median | 15.6 | −1.9 ± 20.0 | 20 / 39 / 56 |
| SBP | Feature model | 13.1 | −0.2 ± 17.0 | 25 / 47 / 65 |
| SBP | Waveform model | 13.0 | −0.1 ± 16.9 | 25 / 48 / 66 |
| DBP | Training median | 10.0 | −0.7 ± 12.7 | 31 / 58 / 77 |
| DBP | Feature model | 8.5 | +0.1 ± 10.9 | 37 / 66 / 85 |
| DBP | Waveform model | 8.5 | +0.5 ± 10.9 | 37 / 67 / 84 |
| MBP | Training median | 12.1 | −1.0 ± 15.4 | 25 / 49 / 69 |
| MBP | Feature model | 9.7 | +0.1 ± 12.6 | 33 / 61 / 79 |
| MBP | Waveform model | 9.6 | +0.6 ± 12.5 | 34 / 61 / 80 |

## Table S5. Overall patient-level coverage (%) of the prediction intervals

Intervals were calibrated on the VitalDB calibration patients and, for MIMIC-III, applied unchanged. Numbers of patients in parentheses.

| Dataset | Target | Model | Method | Nominal 80% (95% CI) | Nominal 90% (95% CI) | Nominal 95% (95% CI) | Median width at 90% (mmHg) | Patients below 70% at 90% (%) |
|---|---|---|---|---|---|---|---|---|
| VitalDB test (627) | SBP | Feature | Split conformal | 80.4 (79.0–81.8) | 90.2 (89.1–91.2) | 94.9 (94.2–95.6) | 55.7 | 6.9 |
| VitalDB test (627) | SBP | Feature | CQR | 80.5 (79.1–81.9) | 90.3 (89.3–91.2) | 95.1 (94.4–95.8) | 53.0 | 6.5 |
| VitalDB test (627) | SBP | Feature | Group-wise (predicted SBP) | 82.0 (80.6–83.3) | 91.3 (90.3–92.2) | 95.8 (95.2–96.4) | 53.4 | 5.4 |
| VitalDB test (627) | SBP | Feature | Group-wise (signal quality) | 81.1 (79.7–82.4) | 90.6 (89.5–91.6) | 95.3 (94.6–96.0) | 53.8 | 6.4 |
| VitalDB test (627) | SBP | Feature | Group-wise (both) | 82.5 (81.1–83.8) | 91.7 (90.7–92.6) | 96.1 (95.5–96.6) | 58.2 | 5.1 |
| VitalDB test (627) | SBP | Feature | CQR + group-wise (both) | 82.7 (81.4–84.0) | 92.1 (91.1–92.9) | 96.4 (95.8–96.9) | 56.1 | 4.8 |
| VitalDB test (627) | SBP | Feature | Split, equal weight per segment | 79.7 (78.2–81.0) | 89.5 (88.5–90.6) | 94.4 (93.6–95.1) | 54.4 | 7.7 |
| VitalDB test (627) | SBP | Waveform | Split conformal | 80.4 (78.8–81.9) | 90.1 (88.9–91.2) | 95.0 (94.3–95.7) | 54.9 | 8.3 |
| VitalDB test (627) | SBP | Waveform | CQR | 80.4 (78.9–81.9) | 90.1 (88.9–91.2) | 95.1 (94.3–95.8) | 52.0 | 7.8 |
| VitalDB test (627) | SBP | Waveform | Group-wise (predicted SBP) | 82.0 (80.4–83.5) | 91.4 (90.3–92.4) | 95.8 (95.1–96.5) | 52.6 | 6.1 |
| VitalDB test (627) | SBP | Waveform | Group-wise (signal quality) | 80.6 (79.1–82.1) | 90.3 (89.2–91.4) | 95.2 (94.5–95.9) | 52.7 | 7.5 |
| VitalDB test (627) | SBP | Waveform | Group-wise (both) | 82.1 (80.6–83.6) | 91.5 (90.4–92.5) | 95.9 (95.2–96.5) | 56.2 | 6.4 |
| VitalDB test (627) | SBP | Waveform | CQR + group-wise (both) | 82.2 (80.6–83.5) | 91.7 (90.6–92.6) | 96.1 (95.5–96.7) | 55.1 | 5.9 |
| VitalDB test (627) | SBP | Waveform | Split, equal weight per segment | 79.5 (77.9–81.0) | 89.3 (88.1–90.4) | 94.4 (93.6–95.2) | 53.5 | 9.4 |
| VitalDB test (627) | DBP | Feature | Split conformal | 78.6 (76.9–80.1) | 89.1 (87.9–90.1) | 94.4 (93.6–95.1) | 34.5 | 9.7 |
| VitalDB test (627) | DBP | Feature | CQR | 78.5 (76.8–80.1) | 88.8 (87.5–90.0) | 94.2 (93.3–95.0) | 33.5 | 11.2 |
| VitalDB test (627) | DBP | Feature | Group-wise (predicted SBP) | 79.7 (77.9–81.1) | 89.9 (88.7–91.0) | 95.1 (94.3–95.7) | 35.3 | 8.1 |
| VitalDB test (627) | DBP | Feature | Group-wise (signal quality) | 79.2 (77.5–80.7) | 89.6 (88.4–90.7) | 94.8 (94.0–95.5) | 33.5 | 9.1 |
| VitalDB test (627) | DBP | Feature | Group-wise (both) | 80.1 (78.4–81.5) | 90.3 (89.1–91.3) | 95.4 (94.6–96.0) | 34.4 | 7.8 |
| VitalDB test (627) | DBP | Feature | CQR + group-wise (both) | 79.9 (78.2–81.3) | 89.9 (88.7–91.0) | 95.1 (94.3–95.9) | 34.5 | 9.9 |
| VitalDB test (627) | DBP | Feature | Split, equal weight per segment | 77.6 (75.8–79.1) | 88.3 (87.0–89.4) | 93.6 (92.7–94.4) | 33.6 | 11.3 |
| VitalDB test (627) | DBP | Waveform | Split conformal | 79.1 (77.3–80.7) | 89.1 (87.8–90.2) | 94.4 (93.5–95.2) | 34.7 | 10.7 |
| VitalDB test (627) | DBP | Waveform | CQR | 78.7 (76.9–80.3) | 89.0 (87.6–90.2) | 94.6 (93.6–95.4) | 33.2 | 11.6 |
| VitalDB test (627) | DBP | Waveform | Group-wise (predicted SBP) | 79.6 (77.9–81.3) | 89.6 (88.2–90.7) | 94.8 (93.9–95.6) | 34.9 | 10.2 |
| VitalDB test (627) | DBP | Waveform | Group-wise (signal quality) | 79.2 (77.5–80.8) | 89.4 (88.2–90.6) | 94.7 (93.9–95.5) | 33.7 | 9.7 |
| VitalDB test (627) | DBP | Waveform | Group-wise (both) | 79.8 (78.0–81.4) | 89.7 (88.4–90.9) | 95.0 (94.1–95.7) | 34.3 | 10.4 |
| VitalDB test (627) | DBP | Waveform | CQR + group-wise (both) | 79.8 (78.0–81.4) | 89.7 (88.4–90.9) | 95.2 (94.3–95.9) | 33.8 | 10.4 |
| VitalDB test (627) | DBP | Waveform | Split, equal weight per segment | 78.0 (76.2–79.6) | 88.2 (86.9–89.4) | 93.7 (92.8–94.6) | 33.8 | 11.5 |
| VitalDB test (627) | MBP | Feature | Split conformal | 79.5 (78.0–80.8) | 89.6 (88.5–90.6) | 94.6 (93.9–95.3) | 40.2 | 8.6 |
| VitalDB test (627) | MBP | Feature | CQR | 79.3 (77.8–80.7) | 89.5 (88.4–90.5) | 94.5 (93.8–95.2) | 38.5 | 8.6 |
| VitalDB test (627) | MBP | Feature | Group-wise (predicted SBP) | 81.0 (79.6–82.2) | 90.9 (89.9–91.8) | 95.6 (94.9–96.2) | 40.7 | 7.0 |
| VitalDB test (627) | MBP | Feature | Group-wise (signal quality) | 80.1 (78.6–81.4) | 90.2 (89.1–91.1) | 95.1 (94.4–95.8) | 38.6 | 7.7 |
| VitalDB test (627) | MBP | Feature | Group-wise (both) | 81.5 (80.1–82.8) | 91.3 (90.3–92.2) | 96.0 (95.4–96.6) | 41.6 | 6.2 |
| VitalDB test (627) | MBP | Feature | CQR + group-wise (both) | 81.5 (80.1–82.8) | 91.2 (90.2–92.1) | 95.9 (95.2–96.5) | 40.5 | 6.7 |
| VitalDB test (627) | MBP | Feature | Split, equal weight per segment | 78.6 (77.1–80.0) | 88.9 (87.8–89.9) | 94.0 (93.3–94.8) | 39.2 | 9.7 |
| VitalDB test (627) | MBP | Waveform | Split conformal | 79.9 (78.2–81.3) | 89.7 (88.5–90.8) | 94.5 (93.7–95.3) | 40.1 | 8.0 |
| VitalDB test (627) | MBP | Waveform | CQR | 80.2 (78.6–81.6) | 89.9 (88.7–91.0) | 94.8 (94.0–95.6) | 38.6 | 8.3 |
| VitalDB test (627) | MBP | Waveform | Group-wise (predicted SBP) | 80.7 (79.1–82.2) | 90.3 (89.1–91.3) | 95.2 (94.5–95.9) | 40.0 | 8.3 |
| VitalDB test (627) | MBP | Waveform | Group-wise (signal quality) | 80.1 (78.5–81.5) | 89.9 (88.7–91.0) | 94.8 (94.0–95.5) | 38.3 | 8.3 |
| VitalDB test (627) | MBP | Waveform | Group-wise (both) | 81.0 (79.4–82.4) | 90.5 (89.4–91.6) | 95.4 (94.7–96.1) | 39.5 | 8.0 |
| VitalDB test (627) | MBP | Waveform | CQR + group-wise (both) | 81.3 (79.8–82.7) | 90.6 (89.5–91.7) | 95.6 (94.8–96.3) | 39.5 | 7.8 |
| VitalDB test (627) | MBP | Waveform | Split, equal weight per segment | 78.9 (77.2–80.4) | 88.9 (87.6–90.0) | 93.9 (93.1–94.7) | 39.1 | 8.9 |
| MIMIC-III (416) | SBP | Feature | Split conformal | 56.9 (53.3–60.6) | 68.9 (65.4–72.6) | 77.5 (74.4–80.9) | 55.7 | 36.3 |
| MIMIC-III (416) | SBP | Feature | CQR | 72.3 (68.9–75.9) | 83.7 (80.9–86.4) | 90.3 (88.2–92.4) | 73.4 | 18.8 |
| MIMIC-III (416) | SBP | Feature | Group-wise (predicted SBP) | 59.1 (55.5–62.8) | 70.8 (67.5–74.3) | 79.5 (76.5–82.5) | 53.4 | 35.6 |
| MIMIC-III (416) | SBP | Feature | Group-wise (signal quality) | 58.9 (55.3–62.6) | 70.9 (67.5–74.5) | 79.7 (76.7–82.8) | 53.8 | 34.1 |
| MIMIC-III (416) | SBP | Feature | Group-wise (both) | 60.5 (56.9–64.2) | 72.5 (69.2–75.9) | 80.9 (78.0–83.8) | 58.2 | 32.7 |
| MIMIC-III (416) | SBP | Feature | CQR + group-wise (both) | 73.7 (70.3–77.3) | 85.2 (82.6–87.8) | 91.8 (89.9–93.7) | 76.1 | 17.5 |
| MIMIC-III (416) | SBP | Feature | Split, equal weight per segment | 56.2 (52.6–59.9) | 67.9 (64.4–71.6) | 76.4 (73.3–79.8) | 54.4 | 37.7 |
| MIMIC-III (416) | SBP | Waveform | Split conformal | 57.9 (54.4–61.6) | 69.7 (66.3–73.3) | 78.3 (75.2–81.5) | 54.9 | 35.8 |
| MIMIC-III (416) | SBP | Waveform | CQR | 64.9 (61.4–68.3) | 76.3 (73.2–79.5) | 83.3 (80.6–86.0) | 61.6 | 29.8 |
| MIMIC-III (416) | SBP | Waveform | Group-wise (predicted SBP) | 60.3 (56.8–64.0) | 72.3 (69.0–75.7) | 80.4 (77.5–83.4) | 52.6 | 33.7 |
| MIMIC-III (416) | SBP | Waveform | Group-wise (signal quality) | 59.5 (56.0–63.2) | 71.3 (68.0–74.8) | 80.0 (77.1–83.0) | 52.7 | 35.1 |
| MIMIC-III (416) | SBP | Waveform | Group-wise (both) | 61.4 (57.9–65.0) | 73.1 (69.8–76.5) | 81.1 (78.2–84.1) | 56.2 | 32.9 |
| MIMIC-III (416) | SBP | Waveform | CQR + group-wise (both) | 66.3 (62.8–69.8) | 77.6 (74.6–80.8) | 84.9 (82.4–87.5) | 65.0 | 26.7 |
| MIMIC-III (416) | SBP | Waveform | Split, equal weight per segment | 57.0 (53.4–60.7) | 68.7 (65.3–72.3) | 77.1 (74.0–80.4) | 53.5 | 37.0 |

Reference without PPG: a split conformal interval around the training median (115.6 mmHg), VitalDB test, nominal 90%: coverage 90.1 (89.2–90.9), median width 65.0 mmHg.

## Table S6. Patient-level coverage (%) of nominal 90% intervals by true pressure (mmHg)

Values are patient-level coverage with 95% bootstrap confidence intervals. Each block starts with the number of patients with at least one segment in the range.

**(a) SBP intervals by true SBP**

| Dataset, model, method | <90 | 90–99 | 100–119 | 120–139 | 140–159 | ≥160 |
|---|---|---|---|---|---|---|
| VitalDB test: patients | 505 | 582 | 624 | 625 | 597 | 475 |
| VitalDB test: feature model, Split conformal | 73.4 (70.6–76.0) | 93.6 (92.4–94.7) | 98.8 (98.3–99.2) | 97.6 (97.0–98.1) | 75.8 (73.8–77.9) | 26.5 (23.9–29.2) |
| VitalDB test: feature model, CQR | 57.9 (55.2–60.5) | 94.7 (93.6–95.7) | 99.8 (99.7–99.9) | 96.7 (95.9–97.4) | 82.8 (80.8–84.7) | 44.3 (41.4–47.3) |
| VitalDB test: feature model, Group-wise (predicted SBP) | 69.9 (67.1–72.5) | 96.2 (95.3–97.0) | 99.5 (99.3–99.7) | 96.8 (96.1–97.4) | 79.3 (77.3–81.2) | 41.6 (38.6–44.6) |
| VitalDB test: feature model, Group-wise (signal quality) | 74.5 (72.0–77.0) | 94.6 (93.5–95.6) | 99.0 (98.5–99.3) | 97.4 (96.8–97.9) | 77.8 (75.8–79.8) | 28.6 (25.9–31.3) |
| VitalDB test: feature model, Group-wise (both) | 72.1 (69.5–74.5) | 96.9 (96.1–97.6) | 99.5 (99.3–99.7) | 96.8 (96.1–97.5) | 80.6 (78.7–82.5) | 44.2 (41.2–47.2) |
| VitalDB test: feature model, CQR + group-wise (both) | 64.1 (61.5–66.6) | 97.2 (96.5–97.8) | 100.0 (99.9–100.0) | 97.4 (96.7–98.0) | 86.1 (84.4–87.8) | 51.4 (48.5–54.4) |
| VitalDB test: patients | 505 | 582 | 624 | 625 | 597 | 475 |
| VitalDB test: waveform model, Split conformal | 76.5 (73.6–79.0) | 92.0 (90.6–93.3) | 97.8 (97.3–98.4) | 96.0 (95.1–96.7) | 79.0 (77.0–81.1) | 37.1 (34.0–40.2) |
| VitalDB test: waveform model, CQR | 63.9 (60.9–66.6) | 94.0 (92.8–95.1) | 98.5 (98.1–98.9) | 93.9 (92.7–95.1) | 84.6 (82.7–86.5) | 59.4 (56.3–62.5) |
| MIMIC-III: patients | 107 | 171 | 280 | 299 | 198 | 126 |
| MIMIC-III: feature model, Split conformal | 32.1 (25.5–39.0) | 77.6 (73.0–82.0) | 96.0 (94.2–97.5) | 92.6 (90.4–94.6) | 46.4 (41.4–51.7) | 6.0 (3.4–9.1) |
| MIMIC-III: feature model, CQR | 47.3 (40.0–55.1) | 95.7 (93.0–97.9) | 100.0 (100.0–100.0) | 99.5 (99.0–99.9) | 86.1 (82.2–89.7) | 39.0 (32.3–46.4) |

**(b) DBP intervals by true DBP**

| Dataset, model, method | <50 | 50–59 | 60–69 | 70–79 | 80–89 | ≥90 |
|---|---|---|---|---|---|---|
| VitalDB test: patients | 521 | 611 | 624 | 607 | 558 | 469 |
| VitalDB test: feature model, Split conformal | 84.0 (82.0–86.0) | 97.7 (97.2–98.2) | 99.4 (99.2–99.6) | 94.4 (93.3–95.3) | 66.8 (64.2–69.2) | 21.0 (18.6–23.5) |
| VitalDB test: feature model, CQR | 80.5 (78.4–82.6) | 99.6 (99.4–99.7) | 98.9 (98.6–99.3) | 93.4 (92.2–94.4) | 77.4 (75.0–79.6) | 39.5 (36.4–42.5) |

## Table S7. Patient-level coverage (%) of nominal 90% SBP intervals in subgroups, VitalDB test patients, feature model

Above / below: share of a patient's segments with the true value above or below the interval, averaged over patients.

| Variable | Group | Patients | Split conformal (95% CI) | Split conformal: above / below | CQR (95% CI) | CQR: above / below |
|---|---|---|---|---|---|---|
| SBP relative to the patient's own median (mmHg) | <-20 | 594 | 79.4 (77.0–81.8) | 0.0 / 20.6 | 68.8 (65.9–71.6) | 0.1 / 31.1 |
| SBP relative to the patient's own median (mmHg) | -20–<20 | 627 | 94.1 (93.0–95.2) | 3.0 / 2.9 | 94.0 (92.9–95.0) | 3.4 / 2.6 |
| SBP relative to the patient's own median (mmHg) | ≥20 | 624 | 68.5 (66.1–70.8) | 31.4 / 0.1 | 75.3 (73.1–77.5) | 24.7 / 0.0 |
| Pulse rate (per minute) | <60 | 475 | 90.5 (88.6–92.2) | 5.3 / 4.2 | 87.0 (84.9–88.9) | 5.8 / 7.2 |
| Pulse rate (per minute) | 60–<80 | 611 | 90.6 (89.3–91.8) | 6.2 / 3.2 | 89.5 (88.2–90.7) | 6.4 / 4.1 |
| Pulse rate (per minute) | 80–<100 | 598 | 80.9 (79.0–82.9) | 14.8 / 4.3 | 85.1 (83.3–86.9) | 11.1 / 3.8 |
| Pulse rate (per minute) | ≥100 | 505 | 71.8 (68.9–74.6) | 21.4 / 6.8 | 79.2 (76.8–81.7) | 14.3 / 6.4 |
| Signal-quality tertile | low | 627 | 86.6 (85.3–87.7) | 8.8 / 4.6 | 89.8 (88.8–90.8) | 6.3 / 3.9 |
| Signal-quality tertile | middle | 624 | 91.3 (90.1–92.4) | 5.7 / 3.0 | 90.2 (89.1–91.3) | 5.8 / 4.1 |
| Signal-quality tertile | high | 618 | 91.7 (90.6–92.8) | 5.4 / 2.9 | 89.5 (88.2–90.6) | 5.9 / 4.6 |
| Age (years) | <40 | 81 | 90.5 (86.9–93.5) | 4.6 / 4.9 | 89.8 (86.5–92.7) | 4.7 / 5.5 |
| Age (years) | 40–<60 | 217 | 91.0 (89.1–92.7) | 5.7 / 3.3 | 90.9 (89.1–92.5) | 5.5 / 3.6 |
| Age (years) | 60–<75 | 232 | 90.1 (88.6–91.6) | 6.8 / 3.1 | 90.0 (88.4–91.4) | 6.3 / 3.7 |
| Age (years) | ≥75 | 97 | 88.2 (85.4–90.7) | 6.9 / 4.9 | 90.1 (87.6–92.2) | 5.4 / 4.5 |
| Sex | F | 292 | 89.7 (88.1–91.2) | 6.1 / 4.2 | 90.0 (88.5–91.4) | 5.3 / 4.7 |
| Sex | M | 335 | 90.6 (89.2–91.9) | 6.2 / 3.2 | 90.6 (89.3–91.8) | 5.9 / 3.5 |
| Body mass index | <18.5 | 52 | 89.4 (85.4–92.4) | 4.0 / 6.6 | 90.8 (87.0–93.8) | 3.1 / 6.1 |
| Body mass index | 18.5–<25 | 396 | 89.5 (88.0–90.9) | 6.4 / 4.1 | 89.7 (88.3–90.9) | 5.8 / 4.5 |
| Body mass index | 25–<30 | 158 | 92.3 (90.9–93.8) | 5.5 / 2.2 | 92.0 (90.3–93.5) | 5.4 / 2.6 |
| Body mass index | ≥30 | 23 | 88.5 (81.9–93.8) † | 10.2 / 1.3 | 86.6 (79.6–92.1) † | 11.0 / 2.4 |
| PPG skewness tertile (planned sensitivity analysis) | low | 626 | 81.0 (79.3–82.7) | 14.4 / 4.6 | 86.6 (85.2–88.0) | 10.4 / 2.9 |
| PPG skewness tertile (planned sensitivity analysis) | middle | 626 | 89.7 (88.6–90.9) | 6.7 / 3.6 | 89.2 (88.0–90.4) | 6.4 / 4.4 |
| PPG skewness tertile (planned sensitivity analysis) | high | 621 | 92.0 (90.8–93.0) | 4.3 / 3.7 | 89.0 (87.7–90.2) | 4.5 / 6.5 |

† Fewer than 30 patients; reported for reference only.
Two patients had two operations with body mass index in different groups and appear in both.

## Table S8. Patient-level coverage (%) of nominal 90% SBP intervals by true and by predicted SBP, VitalDB test patients, feature model

Shrinkage model: coverage expected if the estimate equals a + b × true SBP plus a normal error (Section S1.10). Group-wise: calibrated within bands of predicted SBP.

| SBP band (mmHg) | True SBP: patients | True SBP: observed, split | True SBP: shrinkage model, split | Predicted SBP: patients | Predicted SBP: split (95% CI) | Predicted SBP: group-wise (95% CI) |
|---|---|---|---|---|---|---|
| <90 | 505 | 73.4 | 70.1 | 58 | 92.8 (85.9–98.1) | 90.9 (83.6–96.9) |
| 90–99 | 582 | 93.6 | 90.8 | 472 | 94.8 (93.3–96.2) | 89.7 (87.6–91.6) |
| 100–119 | 624 | 98.8 | 98.6 | 626 | 92.0 (91.0–92.9) | 90.7 (89.7–91.7) |
| 120–139 | 625 | 97.6 | 97.1 | 627 | 84.1 (82.5–85.7) | 89.3 (88.0–90.5) |
| 140–159 | 597 | 75.8 | 74.6 | 522 | 72.9 (70.0–75.8) | 87.0 (84.8–89.1) |
| ≥160 | 475 | 26.5 | 20.2 | 186 | 65.8 (59.9–71.6) | 82.8 (78.0–87.5) |

## Table S10. Recalibration with MIMIC-III patients, feature model, nominal 90% SBP intervals (200 random draws)

Offset corrected: estimates and quantiles first shifted by their mean error in the calibration patients (exploratory). At SBP ≥ 160 mmHg, two measures are shown: the share of such segments covered, pooled over patients, and the patient-level coverage (the mean over patients of each patient's share). With 5 patients the interval is unlimited, so coverage is not shown.

| Method | Offset corrected | Calibration patients | Patient-level coverage, mean (5th–95th percentile) | Draws below 90% (%) | Segments covered at SBP ≥ 160 mmHg (%) | Patient-level coverage at SBP ≥ 160 mmHg (%) | Median width (mmHg) |
|---|---|---|---|---|---|---|---|
| CQR | no | 5 | — | — | — | — | unlimited |
| CQR | no | 10 | 95.7 (88.9–99.8) | 9 | 73.3 | 81.2 | 118.7 |
| CQR | no | 20 | 93.4 (85.2–98.9) | 20 | 61.0 | 71.3 | 102.8 |
| CQR | no | 50 | 91.1 (85.6–96.4) | 35 | 48.2 | 60.7 | 92.6 |
| CQR | no | 100 | 90.9 (86.6–94.6) | 31 | 45.7 | 58.9 | 90.9 |
| CQR | yes | 5 | — | — | — | — | unlimited |
| CQR | yes | 10 | 94.6 (86.8–99.5) | 17 | 80.3 | 86.6 | 112.3 |
| CQR | yes | 20 | 92.3 (82.8–98.4) | 30 | 74.0 | 81.8 | 99.0 |
| CQR | yes | 50 | 90.6 (85.1–95.7) | 42 | 67.5 | 77.0 | 90.0 |
| CQR | yes | 100 | 90.6 (86.0–94.1) | 37 | 67.8 | 77.6 | 88.9 |
| Split conformal | no | 5 | — | — | — | — | unlimited |
| Split conformal | no | 10 | 95.7 (88.6–99.8) | 7 | 69.6 | 78.9 | 134.6 |
| Split conformal | no | 20 | 93.2 (84.3–98.6) | 23 | 54.5 | 66.6 | 114.5 |
| Split conformal | no | 50 | 91.3 (85.5–96.1) | 28 | 42.7 | 56.1 | 103.5 |
| Split conformal | no | 100 | 91.0 (87.0–94.4) | 33 | 39.6 | 53.5 | 101.1 |
| Split conformal | yes | 5 | — | — | — | — | unlimited |
| Split conformal | yes | 10 | 94.7 (86.2–99.7) | 14 | 73.4 | 81.9 | 123.0 |
| Split conformal | yes | 20 | 92.3 (82.1–98.6) | 28 | 63.2 | 74.1 | 106.5 |
| Split conformal | yes | 50 | 90.8 (85.6–95.2) | 37 | 55.3 | 68.3 | 97.4 |
| Split conformal | yes | 100 | 90.7 (86.6–94.3) | 36 | 54.7 | 68.0 | 95.8 |

## Table S11. Withholding SBP estimates with wide CQR intervals, feature model

Thresholds were chosen on the VitalDB calibration patients to keep 10%, 20%, …, 100% of their segments. Values are not shown when fewer than 1% of segments were kept.

| Dataset | Width threshold (mmHg) | Segments kept (%) | MAE (mmHg) | SD of error (mmHg) | Coverage of kept (%) | SD of true SBP (mmHg) |
|---|---|---|---|---|---|---|
| VitalDB test | 43.6 | 10.3 | 10.2 | 12.8 | 85.7 | 14.3 |
| VitalDB test | 46.5 | 19.9 | 10.4 | 13.1 | 87.1 | 15.0 |
| VitalDB test | 48.8 | 29.2 | 10.7 | 13.5 | 87.2 | 15.5 |
| VitalDB test | 51.0 | 38.8 | 10.9 | 13.8 | 88.2 | 16.0 |
| VitalDB test | 53.3 | 48.6 | 11.1 | 14.1 | 88.7 | 16.3 |
| VitalDB test | 56.0 | 58.4 | 11.3 | 14.4 | 89.1 | 16.8 |
| VitalDB test | 59.0 | 68.0 | 11.6 | 14.7 | 89.5 | 17.2 |
| VitalDB test | 63.2 | 77.9 | 11.9 | 15.1 | 89.9 | 17.8 |
| VitalDB test | 70.3 | 88.2 | 12.3 | 15.7 | 90.2 | 18.5 |
| VitalDB test | 144.5 | 100.0 | 13.1 | 17.0 | 90.3 | 20.0 |
| MIMIC-III | 43.6 | 0.0 | — | — | — | — |
| MIMIC-III | 46.5 | 0.1 | — | — | — | — |
| MIMIC-III | 48.8 | 0.3 | — | — | — | — |
| MIMIC-III | 51.0 | 0.8 | — | — | — | — |
| MIMIC-III | 53.3 | 1.8 | 14.9 | 17.7 | 78.5 | 20.1 |
| MIMIC-III | 56.0 | 4.4 | 16.8 | 19.3 | 80.6 | 21.9 |
| MIMIC-III | 59.0 | 9.5 | 17.9 | 20.0 | 80.3 | 22.4 |
| MIMIC-III | 63.2 | 19.1 | 18.3 | 20.3 | 79.5 | 22.0 |
| MIMIC-III | 70.3 | 39.3 | 19.3 | 22.3 | 81.4 | 23.2 |
| MIMIC-III | 144.5 | 100.0 | 21.7 | 26.8 | 83.7 | 25.6 |

## Table S12. Differences between methods and models, nominal 90% SBP intervals

Patient-level coverage (%). Confidence intervals from 2,000 bootstrap resamples of patients; in (a), each patient's pair of values is kept together.

**(a) Paired differences between methods, feature model**

| Dataset | Subgroup | Comparison | Patients | Difference (percentage points, 95% CI) |
|---|---|---|---|---|
| VitalDB test | SBP ≥ 160 | CQR − split | 475 | 17.9 (15.9 to 20.0) |
| VitalDB test | SBP ≥ 160 | Group-wise (both) − split | 475 | 17.7 (16.2 to 19.4) |
| VitalDB test | SBP ≥ 160 | CQR + group-wise − CQR | 475 | 7.1 (6.2 to 8.1) |
| MIMIC-III | all | CQR − split | 416 | 14.7 (12.6 to 16.8) |

**(b) Exploratory alternatives, feature model (not planned in advance)**

Group-wise (PPG pulse rate): calibrated within bands of the pulse rate measured from the PPG (<60, 60–79, 80–99, ≥100 per minute). Shrinkage corrected: the estimate was mapped back by (estimate − a)/b with a and b from the calibration patients (Section S1.10), and a split conformal interval was built around it. Width is the median over segments; for group-wise calibration it is that of all segments.

| Method | Dataset | All (95% CI) | True SBP <90 (95% CI) | True SBP ≥160 (95% CI) | Median width (mmHg) |
|---|---|---|---|---|---|
| Split conformal | VitalDB test | 90.2 (89.1–91.2) | 73.4 (70.6–76.0) | 26.5 (23.9–29.2) | 55.7 |
| Split conformal | MIMIC-III | 68.9 (65.4–72.6) | 32.1 (25.5–39.0) | 6.0 (3.4–9.1) | 55.7 |
| Group-wise (PPG pulse rate) | VitalDB test | 92.9 (92.1–93.8) | 80.9 (78.7–82.9) | 42.1 (39.2–44.9) | 56.4 |
| Group-wise (PPG pulse rate) | MIMIC-III | 77.9 (74.8–81.2) | 57.9 (50.9–65.1) | 21.9 (16.6–27.8) | 70.6 |
| Shrinkage corrected, split | VitalDB test | 90.4 (89.4–91.4) | 91.6 (90.1–93.1) | 76.1 (73.6–78.3) | 102.7 |
| Shrinkage corrected, split | MIMIC-III | 79.3 (76.7–82.0) | 65.2 (58.3–71.9) | 57.7 (50.7–64.6) | 102.7 |

**(c) Waveform model − feature model (exploratory, not planned in advance)**

The same patients for both models. MAE is over segments, with patients resampled; coverage is per patient, with each patient's pair of values kept together. Differences in coverage are in percentage points.

| Dataset | Subgroup | Measure | Patients | Feature model | Waveform model | Difference (95% CI) |
|---|---|---|---|---|---|---|
| VitalDB test | all | MAE (mmHg) | 627 | 13.1 | 13.0 | −0.13 (−0.29 to 0.04) |
| VitalDB test | SBP ≥ 160 | Coverage, split (%) | 475 | 26.5 | 37.1 | 10.7 (8.9 to 12.5) |
| VitalDB test | SBP < 90 | Coverage, split (%) | 505 | 73.4 | 76.5 | 3.1 (1.7 to 4.3) |
| VitalDB test | SBP ≥ 160 | Coverage, CQR (%) | 475 | 44.3 | 59.4 | 15.0 (12.9 to 17.2) |
| VitalDB test | SBP < 90 | Coverage, CQR (%) | 505 | 57.9 | 63.9 | 6.0 (4.4 to 7.7) |
| MIMIC-III | all | Coverage, split (%) | 416 | 68.9 | 69.7 | 0.8 (−0.7 to 2.3) |
| MIMIC-III | all | Coverage, CQR (%) | 416 | 83.7 | 76.3 | −7.4 (−9.1 to −5.6) |

## Table S13. Additional exploratory analyses, nominal 90% SBP intervals, feature model unless stated

Patient-level coverage (%) with 95% bootstrap confidence intervals unless stated. None of these analyses was planned in advance.

**(a) Possible overshoot of the arterial line: segments with a pulse pressure above 100 mmHg, by true SBP (mmHg), removed from calibration and test**

| Analysis | <90 | 90–99 | 100–119 | 120–139 | 140–159 | ≥160 |
|---|---|---|---|---|---|---|
| Segments with pulse pressure > 100 mmHg (%) | 0.0 | 0.0 | 0.0 | 0.0 | 0.9 | 17.9 |
| Split conformal, without these segments | 72.3 (69.6–74.9) | 93.2 (92.0–94.4) | 98.7 (98.3–99.1) | 97.3 (96.8–97.9) | 75.0 (73.0–77.1) | 27.4 (24.6–30.2) |
| CQR, without these segments | 56.5 (53.8–59.1) | 94.3 (93.2–95.3) | 99.8 (99.7–99.9) | 96.5 (95.7–97.2) | 82.2 (80.3–84.1) | 46.5 (43.6–49.8) |

**(b) Variation with the calibration patients: the 1,254 calibration and test patients split again at random into two halves 200 times; mean (2.5th–97.5th percentile) over the splits**

| Method | All | True SBP ≥160 | Predicted SBP 140–159 | Predicted SBP ≥160 |
|---|---|---|---|---|
| Split conformal | 90.2 (88.9–91.4) | 27.5 (25.0–30.1) | 74.4 (72.0–77.1) | 68.3 (64.2–72.5) |
| CQR | 90.2 (88.8–91.5) | 44.8 (42.0–47.7) | 83.6 (81.5–85.9) | 83.9 (80.9–86.9) |
| Group-wise (predicted SBP) | 91.5 (90.2–92.6) | 44.2 (40.5–47.7) | 90.0 (87.4–92.7) | 90.5 (84.5–94.7) |

**(c) One reference reading per patient: the mean of reference minus estimate over each patient's first segments (10 s, 1 min or 5 min) added to the patient's later estimates; intervals calibrated on the corrected calibration patients**

| Reference | Dataset | Method | MAE (mmHg) | All | True SBP <90 | True SBP ≥160 | Median width (mmHg) |
|---|---|---|---|---|---|---|---|
| 10 s | VitalDB test | Split conformal | 19.7 | 89.1 (87.3–90.8) | 80.6 (77.4–83.9) | 62.7 (59.3–66.1) | 82.6 |
| 10 s | VitalDB test | CQR | 19.7 | 89.1 (87.3–90.8) | 77.7 (74.0–81.2) | 73.1 (69.9–76.1) | 80.1 |
| 10 s | MIMIC-III | Split conformal | 13.8 | 94.9 (93.3–96.3) | 81.6 (74.3–88.1) | 87.6 (82.1–92.5) | 82.6 |
| 10 s | MIMIC-III | CQR | 13.8 | 96.9 (95.6–98.1) | 87.1 (80.4–92.8) | 95.7 (92.5–98.7) | 100.5 |
| 1 min | VitalDB test | Split conformal | 18.2 | 90.1 (88.4–91.5) | 78.5 (75.3–81.7) | 61.1 (57.5–64.7) | 78.3 |
| 1 min | VitalDB test | CQR | 18.2 | 90.4 (88.8–91.9) | 75.4 (72.0–78.7) | 73.4 (70.0–76.5) | 76.8 |
| 1 min | MIMIC-III | Split conformal | 12.6 | 95.0 (93.4–96.3) | 79.3 (71.5–86.4) | 85.6 (80.0–91.1) | 78.3 |
| 1 min | MIMIC-III | CQR | 12.6 | 97.1 (95.8–98.2) | 84.8 (77.7–91.0) | 95.7 (92.2–98.7) | 97.0 |
| 5 min | VitalDB test | Split conformal | 15.1 | 91.1 (89.8–92.3) | 75.1 (72.0–78.4) | 54.2 (50.4–57.7) | 66.2 |
| 5 min | VitalDB test | CQR | 15.1 | 91.2 (89.8–92.4) | 67.0 (63.5–70.6) | 68.6 (65.2–72.0) | 64.6 |
| 5 min | MIMIC-III | Split conformal | 11.4 | 93.9 (92.3–95.5) | 79.2 (71.1–86.7) | 80.1 (73.1–86.5) | 66.2 |
| 5 min | MIMIC-III | CQR | 11.4 | 96.9 (95.7–98.0) | 83.5 (75.9–90.5) | 93.3 (88.4–97.4) | 84.9 |

**(d) Weighting with unlabelled MIMIC-III data: domain classifiers on fewer features or with the density ratio clipped to 0.1–10 (main analysis: all 42 features, ratio clipped to 0.001–1000)**

Effective calibration patients: Kish effective sample size of the calibration patients, with the weights summed per patient, as a share of all calibration patients. Without weighting, coverage on MIMIC-III was 68.9% (split conformal) and 83.7% (CQR).

| Classifier | AUC | Effective calibration patients (%) | Method | Coverage on MIMIC-III | Unlimited intervals (%) | Median width (mmHg) |
|---|---|---|---|---|---|---|
| All 42 features (main analysis) | 0.997 | 10.7 | Split conformal | 99.1 (98.6–99.5) | 83.4 | unlimited |
| All 42 features (main analysis) | 0.997 | 10.7 | CQR | 98.5 (97.8–99.1) | 83.4 | unlimited |
| Without b/a and upslope time | 0.995 | 11.6 | Split conformal | 98.8 (98.2–99.3) | 78.8 | unlimited |
| Without b/a and upslope time | 0.995 | 11.6 | CQR | 98.3 (97.5–98.9) | 78.8 | unlimited |
| Rhythm features only | 0.734 | 58.4 | Split conformal | 74.8 (71.6–78.2) | 0.0 | 63.5 |
| Rhythm features only | 0.734 | 58.4 | CQR | 83.8 (81.1–86.5) | 0.0 | 73.4 |
| All features, ratio clipped to 0.1–10 | 0.997 | 40.9 | Split conformal | 90.3 (88.2–92.3) | 0.0 | 104.7 |
| All features, ratio clipped to 0.1–10 | 0.997 | 40.9 | CQR | 92.9 (91.1–94.6) | 0.0 | 99.4 |

**(e) Waveform model trained with five other seeds: SBP accuracy in the 376 validation patients**

| Run | MAE (mmHg) | Mean error ± SD (mmHg) |
|---|---|---|
| main | 13.5 | −0.3 ± 17.5 |
| seed1 | 13.4 | −0.4 ± 17.4 |
| seed2 | 13.4 | −0.6 ± 17.3 |
| seed3 | 13.4 | −2.0 ± 17.3 |
| seed4 | 13.3 | −0.7 ± 17.3 |
| seed5 | 13.4 | −1.0 ± 17.3 |

## Table S14. Mean absolute error with 95% confidence intervals, VitalDB test patients (mmHg)

Confidence intervals from 2,000 bootstrap resamples of patients.

| Target | Model | MAE (95% CI) |
|---|---|---|
| SBP | Training median | 15.6 (15.3–16.0) |
| SBP | Feature model | 13.1 (12.7–13.6) |
| SBP | Waveform model | 13.0 (12.5–13.5) |
| DBP | Training median | 10.0 (9.7–10.4) |
| DBP | Feature model | 8.5 (8.2–8.8) |
| DBP | Waveform model | 8.5 (8.1–8.8) |
| MBP | Training median | 12.1 (11.8–12.4) |
| MBP | Feature model | 9.7 (9.4–10.0) |
| MBP | Waveform model | 9.6 (9.2–10.0) |

## Table S15. Nominal 90% intervals sorted against 140 mmHg (true SBP ≥ 140 mmHg) and 65 mmHg (true MBP < 65 mmHg)

Shares of segments, averaged first within each patient. Exploratory.

| Dataset | Model | Method | SBP ≥ 140: wholly ≥ 140 (%) | SBP ≥ 140: contains 140 (%) | SBP ≥ 140: wholly < 140 (%) | MBP < 65: wholly < 65 (%) | MBP < 65: contains 65 (%) | MBP < 65: wholly ≥ 65 (%) |
|---|---|---|---|---|---|---|---|---|
| VitalDB test | Feature | Split conformal | 1.0 | 93.5 | 5.5 | 0.0 | 91.4 | 8.6 |
| VitalDB test | Feature | CQR | 0.0 | 95.4 | 4.6 | 0.0 | 93.8 | 6.2 |
| VitalDB test | Waveform | Split conformal | 1.8 | 92.0 | 6.3 | 0.0 | 91.0 | 9.0 |
| VitalDB test | Waveform | CQR | 0.0 | 94.0 | 6.0 | 0.0 | 93.1 | 6.9 |
| MIMIC-III | Feature | Split conformal | 0.0 | 75.0 | 25.0 | 0.0 | 60.1 | 39.9 |
| MIMIC-III | Feature | CQR | 0.0 | 98.1 | 1.9 | 0.0 | 88.6 | 11.4 |
| MIMIC-III | Waveform | Split conformal | 0.0 | 77.8 | 22.2 | 0.0 | 56.4 | 43.6 |
| MIMIC-III | Waveform | CQR | 0.0 | 83.5 | 16.5 | 0.0 | 76.1 | 23.9 |

## Table S16. Distribution of the true SBP and of the coverage of each patient

Nominal 90% SBP intervals, feature model. Aggregate shares only.

**(a) True SBP, share of segments per 5 mmHg**

| True SBP (mmHg) | VitalDB test (% of segments) | MIMIC-III (% of segments) |
|---|---|---|
| 40–<45 | 0.00 | 0.00 |
| 45–<50 | 0.00 | 0.00 |
| 50–<55 | 0.00 | 0.00 |
| 55–<60 | 0.01 | 0.02 |
| 60–<65 | 0.04 | 0.15 |
| 65–<70 | 0.12 | 0.32 |
| 70–<75 | 0.33 | 0.67 |
| 75–<80 | 0.77 | 1.33 |
| 80–<85 | 1.76 | 1.66 |
| 85–<90 | 3.23 | 2.75 |
| 90–<95 | 5.29 | 3.62 |
| 95–<100 | 7.33 | 4.65 |
| 100–<105 | 9.36 | 6.08 |
| 105–<110 | 9.99 | 7.13 |
| 110–<115 | 10.21 | 8.41 |
| 115–<120 | 9.97 | 8.70 |
| 120–<125 | 9.23 | 7.75 |
| 125–<130 | 7.88 | 7.64 |
| 130–<135 | 6.73 | 7.26 |
| 135–<140 | 5.16 | 5.84 |
| 140–<145 | 3.84 | 4.75 |
| 145–<150 | 2.76 | 4.60 |
| 150–<155 | 1.90 | 3.47 |
| 155–<160 | 1.30 | 2.95 |
| 160–<165 | 0.90 | 2.40 |
| 165–<170 | 0.62 | 2.03 |
| 170–<175 | 0.38 | 1.54 |
| 175–<180 | 0.27 | 1.43 |
| 180–<185 | 0.20 | 1.38 |
| 185–<190 | 0.14 | 0.64 |
| 190–<195 | 0.09 | 0.17 |
| 195–<200 | 0.05 | 0.22 |
| 200–<205 | 0.04 | 0.16 |
| 205–<210 | 0.02 | 0.06 |
| 210–<215 | 0.02 | 0.06 |
| 215–<220 | 0.01 | 0.08 |
| 220–<225 | 0.01 | 0.02 |
| 225–<230 | 0.01 | 0.01 |
| 230–<235 | 0.00 | 0.01 |
| 235–<240 | 0.00 | 0.01 |
| 240–<245 | 0.00 | 0.01 |
| 245–<250 | 0.00 | 0.01 |
| 250–<255 | 0.00 | 0.01 |
| 255–<260 | 0.00 | 0.01 |
| 260–<265 | 0.00 | 0.00 |
| 265–<270 | 0.00 | 0.00 |
| 270–<275 | 0.00 | 0.00 |
| 275–<280 | 0.00 | 0.00 |
| 280–<285 | 0.00 | 0.00 |
| 285–<290 | 0.00 | 0.00 |
| 290–<295 | 0.00 | 0.00 |
| 295–300 | 0.00 | 0.00 |

**(b) Coverage of each patient, share of patients per 5 percentage points**

| Coverage of the patient (%) | VitalDB test, Split conformal (% of patients) | VitalDB test, CQR (% of patients) | MIMIC-III, Split conformal (% of patients) | MIMIC-III, CQR (% of patients) |
|---|---|---|---|---|
| 0–<5 | 0.0 | 0.0 | 11.3 | 4.1 |
| 5–<10 | 0.0 | 0.0 | 2.4 | 1.7 |
| 10–<15 | 0.5 | 0.0 | 1.9 | 1.0 |
| 15–<20 | 0.3 | 0.3 | 2.2 | 0.7 |
| 20–<25 | 0.0 | 0.6 | 2.2 | 1.0 |
| 25–<30 | 0.0 | 0.0 | 1.4 | 1.7 |
| 30–<35 | 0.2 | 0.3 | 2.2 | 1.4 |
| 35–<40 | 0.5 | 0.2 | 1.2 | 0.2 |
| 40–<45 | 0.8 | 0.3 | 1.7 | 1.4 |
| 45–<50 | 0.5 | 0.3 | 1.9 | 1.4 |
| 50–<55 | 0.3 | 0.3 | 2.2 | 1.7 |
| 55–<60 | 1.1 | 0.8 | 2.2 | 1.0 |
| 60–<65 | 1.3 | 2.2 | 1.2 | 0.7 |
| 65–<70 | 1.4 | 1.1 | 2.4 | 0.7 |
| 70–<75 | 2.2 | 2.2 | 3.1 | 1.9 |
| 75–<80 | 4.5 | 5.3 | 4.6 | 3.8 |
| 80–<85 | 6.2 | 4.8 | 4.3 | 2.4 |
| 85–<90 | 10.0 | 10.5 | 4.3 | 3.1 |
| 90–<95 | 21.9 | 22.8 | 7.0 | 5.8 |
| 95–100 | 48.3 | 47.8 | 40.4 | 64.2 |
