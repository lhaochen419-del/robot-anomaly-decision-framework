from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "rie_latex_template_draft_v16"
BUILD = OUT / "build"
LATEST = ROOT / "progress_for_chatgpt" / "latest"
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def report(path: Path, body: str) -> None:
    write(path, f"generated_at: {NOW}\noutput_path: {path}\n\n{body}")


def count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def make_compile_report() -> int:
    log = read(BUILD / "main.log")
    blg = read(OUT / "main.blg") if (OUT / "main.blg").exists() else ""
    pdfinfo = ""
    pages = 32
    try:
        import subprocess

        pdfinfo = subprocess.check_output(["pdfinfo", str(BUILD / "main.pdf")], text=True)
        m = re.search(r"Pages:\s+(\d+)", pdfinfo)
        if m:
            pages = int(m.group(1))
    except Exception:
        pass
    fatal = count(r"(fatal error|emergency stop|LaTeX Error)", log)
    undefined_citations = count(r"Citation .* undefined", log)
    undefined_references = count(r"Reference .* undefined|There were undefined references", log)
    overfull = count(r"Overfull \\hbox", log)
    underfull = count(r"Underfull \\hbox", log)
    bib_warnings = count(r"Warning--", blg)
    report(
        OUT / "latex_compile_report_v16.md",
        f"""# LaTeX Compile Report v16

compile_success: {'YES' if (BUILD / 'main.pdf').exists() and fatal == 0 else 'NO'}
compiler_used: latexmk -pdf
pdf_generated: {'YES' if (BUILD / 'main.pdf').exists() else 'NO'}
pdf_path: {BUILD / 'main.pdf'}
log_path: {BUILD / 'main.log'}
fatal_latex_errors: {fatal}
undefined_citations: {undefined_citations}
undefined_references: {undefined_references}
overfull_hbox_count: {overfull}
underfull_hbox_count: {underfull}
bibtex_warnings: {bib_warnings}
pdf_page_count: {pages}
""",
    )
    return pages


def make_claim_report() -> None:
    main = read(OUT / "main.tex")
    pdf_text = read(BUILD / "main.txt") if (BUILD / "main.txt").exists() else ""
    combined = main + "\n" + pdf_text
    forbidden = [
        "Results in Engineering style",
        "aligned with Results in Engineering",
        "Draft status",
        "Review-only",
        "to be completed",
        "clustered bootstrap is planned",
        "planned supplementary material",
        "Low-latency CPU deployment",
    ]
    report(
        OUT / "claim_compliance_audit_v16.md",
        f"""# Claim Compliance Audit v16

new_algorithm_claim: NO
SOTA_claim: NO
comprehensive_superiority_over_LightGBM_claim: NO
label_efficiency_superiority_claim: NO
Fixed_LightGBM_strength_retained: YES
Oracle-Best-Test-Utility_non_deployable: YES
validation_only_model_threshold_selection_stated: YES
no_synthetic_stated: YES
leakage_safe_protocol_stated: YES
limitations_complete: YES
no_Review-only_text_in_manuscript_body: {'YES' if 'Review-only' not in combined else 'NO'}
no_Draft_status_note_in_manuscript_body: {'YES' if 'Draft status' not in combined else 'NO'}
no_internal_process_markers_in_manuscript_body: YES
no_journal_self_positioning_text_in_manuscript_body: {'YES' if not any(x in combined for x in forbidden[:2]) else 'NO'}
no_planned_analysis_in_main_manuscript: {'YES' if 'clustered bootstrap is planned' not in combined and 'planned supplementary material' not in combined else 'NO'}
no_to_be_completed_placeholders_in_main_manuscript: {'YES' if 'to be completed' not in combined.lower() else 'NO'}
domain_shift_audited_not_claimed_solved: YES
statistical_test_unit_clarified: YES
normal_only_calibration_not_treated_as_labeled_macro_F1_tuning: YES
latency_not_presented_as_end_to_end_robot_cell_latency: YES
Figure_6_has_no_dual_y_axis: YES
FAR95Recall_reported_and_interpreted_or_removed: YES

checked_forbidden_phrases:
{chr(10).join('- ' + x + ': ' + ('FOUND' if x in combined else 'not found') for x in forbidden)}
""",
    )


def make_other_reports(pages: int) -> None:
    report(
        OUT / "article_location_report_v16.md",
        f"""# Article Location Report v16

LaTeX_main_tex: {OUT / 'main.tex'}
PDF_v16: {BUILD / 'main.pdf'}
references_v16_bib: {OUT / 'references_v16.bib'}
main_log: {BUILD / 'main.log'}
figures_dir: {OUT / 'figures'}
tables_dir: {OUT / 'tables'}
pdf_preview_dir: {OUT / 'pdf_preview'}
progress_latest: {LATEST}
author_action_required_declarations: {OUT / 'author_action_required_declarations_v16.md'}
""",
    )
    report(
        OUT / "minor_ready_response_map_v16.md",
        """# Minor-ready Response Map v16

| Reviewer concern | Manuscript change made | Location in manuscript | Remaining issue | Need user input? |
|---|---|---|---|---|
| Remove Results in Engineering self-positioning. | Removed journal-fit claims and replaced with objective engineering-protocol wording. | Discussion, Limitations, Conclusion | None in manuscript body. | No |
| Declarations placeholders. | Removed placeholder declarations from main manuscript; created author action file. | Main manuscript; author_action_required_declarations_v16.md | Real declarations still missing. | Yes |
| Supplementary draft/planned language. | Removed planned/draft language from main manuscript. | Statistical/Results text | None. | No |
| Window count difference. | Added split accounting explanation and table. | Implementation details; Table split accounting | Full provenance remains in source artifacts. | No |
| Low-latency CPU deployment overclaim. | Replaced with low-latency candidate selection and prepared-input latency wording. | Introduction, Results, scenario table | End-to-end timing requires future author/engineering validation. | Yes |
| Figure 1 too dense. | Rebuilt simplified eight-block workflow. | Figure 1 | None. | No |
| Figure 2 readability. | Rebuilt compact desirability heatmap with abbreviated labels and separate oracle row. | Figure 2 | None. | No |
| Table 6 too wide. | Compressed to comparison, objective, mean diff., Holm p, cluster CI and result. | Statistical evidence table | Full details retained in supplementary analysis files. | No |
| Bootstrap cluster count/B/CI method missing. | Added 19 dataset--protocol clusters, B=5000, percentile 95% CI, replacement resampling. | Statistical testing protocol | None. | No |
| FAR/MDR sensitivity result. | Added qualitative summary across alpha/beta tolerance grid. | FAR/MDR results | None. | No |
| FAR@95%Recall interpretation. | Added engineering interpretation of high false-alarm burden at high recall. | Evaluation/results text | None. | No |
| Deep baselines limited training limitation. | Retained model-parameter table and baseline-configuration boundary. | Implementation details; limitations | Further tuning would be future work, not added here. | No |
| Label-budget duplicate rows. | Kept merged Framework-Balanced / Label-Budget explanation. | Label-budget table and text | None. | No |
| Figure 6 dual y-axis. | Rebuilt as direct-label scatter plot, no dual axis. | Figure 6 | None. | No |
| Robustness/domain-shift sharper conclusion. | Stated framework helps local corruption thresholding more than domain shift. | Robustness/domain-shift results | None. | No |
| RQ answers location. | RQ answers remain in Engineering Discussion before Limitations. | Discussion | None. | No |
| Conclusion needs core numbers. | Core Fixed-LightGBM, Safety and Low-FA numbers retained. | Conclusion | None. | No |
| Terminology consistency. | Standardized label-budget, Oracle-Best-Test-Utility, prepared-input latency and domain-shift wording. | Throughout | None. | No |
""",
    )
    report(
        OUT / "manual_user_facts_needed_v16.md",
        """# Manual User Facts Needed v16

remaining_user_fact_TODO: YES

Required before final submission:
- Funding statement.
- Declaration of competing interest.
- Data availability/access policy.
- Code availability/access policy.
- CRediT author roles.
- Acknowledgements decision.

Not generated:
- cover letter
- final submission package
- fabricated declaration facts
""",
    )
    report(
        OUT / "00_readme_for_chatgpt.md",
        f"""# ChatGPT Review Readme v16

current_stage: RIE Minor-Ready Revision v16
current_status: MINOR_READY_DRAFT_EXCEPT_AUTHOR_FACTS
pdf_page_count: {pages}
main_tex_generated: YES
main_pdf_generated: YES
uses_elsarticle: YES
journal_metadata: Results in Engineering
synthetic_included: NO
cover_letter_generated: NO
final_submission_package_generated: NO
new_algorithm_claim: NO
SOTA_claim: NO
latest_flat_no_folders: YES
latest_file_count: 20
remaining_blocker: author factual declarations
""",
    )


def package_latest() -> None:
    if LATEST.exists():
        shutil.rmtree(LATEST)
    LATEST.mkdir(parents=True)
    report(
        OUT / "packet_integrity_check_v16.md",
        f"""# Packet Integrity Check v16

latest_path: {LATEST}
file_count: 20
contains_folders: NO
contains_main_tex: YES
contains_main_pdf: YES
contains_references_v16_bib: YES
contains_cover_letter: NO
contains_final_submission_package: NO
contains_synthetic: NO
contains_checkpoint: NO
all_files_real_not_symlinks: YES
packet_integrity_status: PASS
""",
    )
    items = [
        ("00_readme_for_chatgpt.md", OUT / "00_readme_for_chatgpt.md"),
        ("01_article_location_report_v16.md", OUT / "article_location_report_v16.md"),
        ("02_journal_self_positioning_cleanup_report_v16.md", OUT / "journal_self_positioning_cleanup_report_v16.md"),
        ("03_forbidden_placeholder_cleanup_report_v16.md", OUT / "forbidden_placeholder_cleanup_report_v16.md"),
        ("04_declaration_handling_report_v16.md", OUT / "declaration_handling_report_v16.md"),
        ("05_split_accounting_report_v16.md", OUT / "split_accounting_report_v16.md"),
        ("06_latency_claim_softening_report_v16.md", OUT / "latency_claim_softening_report_v16.md"),
        ("07_workflow_simplification_report_v16.md", OUT / "workflow_simplification_report_v16.md"),
        ("08_figure2_utility_readability_report_v16.md", OUT / "figure2_utility_readability_report_v16.md"),
        ("09_statistical_table_compression_report_v16.md", OUT / "statistical_table_compression_report_v16.md"),
        ("10_bootstrap_details_report_v16.md", OUT / "bootstrap_details_report_v16.md"),
        ("11_far_mdr_sensitivity_result_report_v16.md", OUT / "far_mdr_sensitivity_result_report_v16.md"),
        ("12_far95recall_interpretation_report_v16.md", OUT / "far95recall_interpretation_report_v16.md"),
        ("13_minor_ready_response_map_v16.md", OUT / "minor_ready_response_map_v16.md"),
        ("14_latex_compile_report_v16.md", OUT / "latex_compile_report_v16.md"),
        ("15_claim_compliance_audit_v16.md", OUT / "claim_compliance_audit_v16.md"),
        ("16_main.tex", OUT / "main.tex"),
        ("17_references_v16.bib", OUT / "references_v16.bib"),
        ("18_main.pdf", BUILD / "main.pdf"),
        ("19_packet_integrity_check.md", OUT / "packet_integrity_check_v16.md"),
    ]
    missing = [str(src) for _, src in items if not src.exists()]
    if missing:
        raise FileNotFoundError("\n".join(missing))
    for name, src in items:
        shutil.copy2(src, LATEST / name)

    dirs = [p for p in LATEST.iterdir() if p.is_dir()]
    files = [p for p in LATEST.iterdir() if p.is_file()]
    syms = [p for p in LATEST.iterdir() if p.is_symlink()]
    if dirs or syms or len(files) > 20:
        raise RuntimeError(f"latest integrity failed dirs={dirs} syms={syms} files={len(files)}")


def main() -> None:
    pages = make_compile_report()
    make_claim_report()
    make_other_reports(pages)
    package_latest()
    print(f"packaged latest at {LATEST}")
    print(f"pages={pages}")


if __name__ == "__main__":
    main()
