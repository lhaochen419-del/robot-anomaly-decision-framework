# Command manifest overview

Run from repository root after placing datasets.

1. Prepare IMAD-DS robotic arm: python scripts/prepare_imadds_raw_windows.py
2. Prepare unified datasets: python scripts/prepare_real_data.py --dataset imadds_robotic_arm
3. Run benchmark aggregation: python scripts/complete_engineering_benchmark_v2.py
4. Validate framework strategies: python scripts/validate_framework_strategy_v1.py
5. Package RIE results: python scripts/package_rie_results_v1.py
6. Rebuild manuscript figures/tables where applicable: python scripts/build_minor_ready_revision_v16.py

Expected outputs are aggregate CSV/MD files under outputs/ and copied result summaries under results_aggregated/.
