from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "rie_latex_template_draft_v16"
DST = ROOT / "outputs" / "rie_latex_template_draft_v17_official_elsarticle"
ELS = ROOT / "elsarticle"
NOW = datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Missing source draft: {SRC}")
    if not ELS.exists():
        raise FileNotFoundError(f"Missing official elsarticle bundle: {ELS}")
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)

    # Put the official bundle files next to main.tex so LaTeX uses them before
    # the system TeX Live copy. The .cls itself is generated from the official
    # .dtx/.ins after this script runs.
    for name in [
        "elsarticle.dtx",
        "elsarticle.ins",
        "elsarticle-num.bst",
        "elsarticle-num-names.bst",
        "elsarticle-harv.bst",
        "elsarticle-template-num.tex",
        "elsarticle-template-num-names.tex",
        "elsarticle-template-harv.tex",
        "README",
        "manifest.txt",
        "changelog.txt",
    ]:
        src = ELS / name
        if src.exists():
            shutil.copy2(src, DST / name)

    main_path = DST / "main.tex"
    text = main_path.read_text(encoding="utf-8")
    begin_doc = text.index(r"\begin{document}")
    body = text[begin_doc:]

    # Keep article content unchanged. Only replace the template/preamble with
    # the official numerical elsarticle template style plus packages needed by
    # the existing manuscript tables and hyperlinks.
    official_preamble = r"""%% 
%% Official Elsevier elsarticle template integration.
%% Based on the local Elsevier Elsarticle Bundle in ../../elsarticle/.
%% Article content below \begin{document} is preserved from the current draft.
%%
\documentclass[preprint,12pt]{elsarticle}

%% Use the option review to obtain double line spacing
%% \documentclass[preprint,review,12pt]{elsarticle}

%% Use the options 1p,twocolumn; 3p; 3p,twocolumn; 5p; or 5p,twocolumn
%% for a journal layout:
%% \documentclass[final,1p,times]{elsarticle}
%% \documentclass[final,1p,times,twocolumn]{elsarticle}
%% \documentclass[final,3p,times]{elsarticle}
%% \documentclass[final,3p,times,twocolumn]{elsarticle}
%% \documentclass[final,5p,times]{elsarticle}
%% \documentclass[final,5p,times,twocolumn]{elsarticle}

%% The amssymb package provides various useful mathematical symbols
\usepackage{amssymb}
%% The amsmath package provides various useful equation environments.
\usepackage{amsmath}

%% Manuscript-specific packages required by existing tables, figures and links.
\usepackage{microtype}
\usepackage{ragged2e}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{xurl}
\usepackage[hypertexnames=false]{hyperref}
\pdfstringdefDisableCommands{%
  \def\corref#1{}%
  \def\cortext#1#2{}%
  \def\ead#1{}%
}
\usepackage{float}

\journal{Results in Engineering}

"""
    new_text = official_preamble + body
    new_text = new_text.replace(r"\bibliography{references_v16}", r"\bibliography{references_v17}")
    main_path.write_text(new_text, encoding="utf-8")

    ref16 = DST / "references_v16.bib"
    if ref16.exists():
        shutil.copy2(ref16, DST / "references_v17.bib")
    elif (DST / "references.bib").exists():
        shutil.copy2(DST / "references.bib", DST / "references_v17.bib")

    report = f"""generated_at: {NOW}
output_path: {DST / 'official_elsarticle_template_switch_report_v17.md'}

# Official elsarticle Template Switch Report v17

source_draft: {SRC}
output_draft: {DST}
official_bundle: {ELS}
official_template_read: elsarticle-template-num.tex
official_readme_read: README
official_manifest_read: manifest.txt
official_changelog_read: changelog.txt
local_cls_expected: {DST / 'elsarticle.cls'}
local_bst_used: {DST / 'elsarticle-num.bst'}
documentclass: \\documentclass[preprint,12pt]{{elsarticle}}
journal: Results in Engineering
bibliography_style: elsarticle-num
article_content_changed: NO
experiment_values_changed: NO
figures_changed: NO
tables_changed: NO
notes: The preamble was replaced with the official Elsevier numerical-template structure plus only the packages needed by the existing manuscript content.
"""
    write(DST / "official_elsarticle_template_switch_report_v17.md", report)


if __name__ == "__main__":
    main()
