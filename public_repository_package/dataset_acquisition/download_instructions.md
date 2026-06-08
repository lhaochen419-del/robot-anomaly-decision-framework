# Real Dataset Download Instructions

## RoAD / Robotic Arm Dataset
- Official source: https://gitlab.com/AlessioMascolini/roaddataset
- Paper DOI: https://doi.org/10.1109/IECON51785.2023.10311726
- Expected access: `pip install git+https://gitlab.com/AlessioMascolini/roaddataset/`
- Expected data object: `RoADDataset.Dataset().sets` with subsets `training`, `collision`, `control`, `weight`, and `velocity`.
- Expected columns: 86 numeric sensor channels; anomaly subsets include channel 87 as anomaly label.
- Place local files, if not using the package, under `robot_cirfl/data/raw/road/`.
- Run: `python scripts/prepare_real_data.py --dataset road`.

## IMAD-DS Robotic Arm
- Official source: https://zenodo.org/records/12665499
- DOI: https://doi.org/10.5281/zenodo.12665499
- Expected files: download `RoboticArm.7z` from Zenodo and extract under `robot_cirfl/data/raw/imadds/RoboticArm/`.
- Expected sensors: analog microphone, 3-axis accelerometer, 3-axis gyroscope.
- Expected labels: normal/abnormal files or label columns depending on extracted format.
- Expected conditions: source/target domain, load/domain shifts, environmental-noise shifts.
- Run: `python scripts/prepare_real_data.py --dataset imadds_robotic_arm`.

## NIST UR Robot Degradation / Health Data
- Official source: https://doi.org/10.18434/M31962
- Landing page: https://www.nist.gov/el/intelligent-systems-division-73500/degradation-measurement-robot-arm-position-accuracy
- Expected files: NIST tables and header files, including `UR5TestResult_header.xlsx` when available.
- Expected channels: TCP health data and UR5 controller-level joint positions, velocities, currents, accelerations, torques, and temperatures.
- Place files under `robot_cirfl/data/raw/nist_ur/`.
- Run: `python scripts/prepare_real_data.py --dataset nist_ur`.

## KUKA LWR4+ Joint Torque Collision/Contact Dataset
- Official source: https://zenodo.org/records/6461868
- Expected files: `cls-joint-1.csv` ... `cls-joint-7.csv`, `ctc-joint-*.csv`, `fre-joint-*.csv`.
- Expected columns: each CSV is N segments by 1024 time samples for one joint and class.
- Expected labels: `fre` normal, `ctc` intentional contact, `cls` accidental collision.
- Place files under `robot_cirfl/data/raw/kuka_torque/`.
- Run: `python scripts/prepare_real_data.py --dataset kuka_torque`.

Synthetic sanity data remains separate and must not be used as paper evidence.
