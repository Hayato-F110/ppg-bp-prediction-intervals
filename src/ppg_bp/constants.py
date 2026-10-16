"""Constants shared by the analyses."""

SBP_HIGH = 160  # mmHg: lower edge of the highest band of true SBP (the extreme high pressures)
SBP_LOW = 90  # mmHg: upper edge of the lowest band of true SBP (the extreme low pressures)
MAIN_LEVEL = 0.9  # nominal coverage of the main analysis (primary_level in configs/conformal.yaml)
FEW_PATIENTS = 30  # strata with fewer patients are reference only (min_subjects_per_stratum in configs/conformal.yaml)
