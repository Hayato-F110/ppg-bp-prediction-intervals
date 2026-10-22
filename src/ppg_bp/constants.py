"""Constants shared by the analyses."""

SBP_HIGH = 160  # mmHg: lower edge of the highest band of true SBP (the extreme high pressures)
SBP_LOW = 90  # mmHg: upper edge of the lowest band of true SBP (the extreme low pressures)
MAIN_LEVEL = 0.9  # nominal coverage of the main analysis (primary_level in configs/conformal.yaml)
FEW_PATIENTS = 30  # strata with fewer patients are reference only (min_subjects_per_stratum in configs/conformal.yaml)
EXTRA_SEEDS = (1, 2, 3, 4, 5)  # further training runs of the waveform model (table S8)
RESPLITS = 200  # random re-splits of the calibration and test patients (exploratory analyses)
# mmHg: bins of the distribution of the true SBP (fig. and table S16); 300 mmHg is the upper limit of rule 2
SBP_BIN_EDGES = tuple(range(40, 301, 5))
COVERAGE_BIN_EDGES = tuple(range(0, 101, 5))  # %: bins of the coverage of each patient (fig. and table S16)
