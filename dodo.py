"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based

"""

#######################################
## Configuration and Helpers for PyDoit
#######################################
## Make sure the src folder is in the path
import sys

sys.path.insert(1, "./src/")

import shutil
from os import environ
from pathlib import Path

from settings import config
from trading_calendar import panel_dates

DOIT_CONFIG = {"backend": "sqlite3", "dep_file": "./.doit-db.sqlite"}


BASE_DIR = config("BASE_DIR")
DATA_DIR = config("DATA_DIR")
MANUAL_DATA_DIR = config("MANUAL_DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")
OS_TYPE = config("OS_TYPE")
USER = config("USER")

# Trading days the panel covers. Derived from the exchange calendar, so doit
# can build the task list before any data exists.
PANEL_DATES = panel_dates()

## Helpers for handling Jupyter Notebook tasks
environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"


# fmt: off
## Helper functions for automatic execution of Jupyter notebooks
def jupyter_execute_notebook(notebook_path):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
def jupyter_to_html(notebook_path, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --output-dir={output_dir} {notebook_path}"
def jupyter_to_md(notebook_path, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f"jupytext --to markdown --output-dir={output_dir} {notebook_path}"
def jupyter_clear_output(notebook_path):
    """Clear the output of a notebook"""
    return f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
# fmt: on


def mv(from_path, to_path):
    """Move a file to a folder"""
    from_path = Path(from_path)
    to_path = Path(to_path)
    to_path.mkdir(parents=True, exist_ok=True)
    if OS_TYPE == "nix":
        command = f"mv {from_path} {to_path}"
    else:
        command = f"move {from_path} {to_path}"
    return command


def copy_file(origin_path, destination_path, mkdir=True):
    """Create a Python action for copying a file."""

    def _copy_file():
        origin = Path(origin_path)
        dest = Path(destination_path)
        if mkdir:
            dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, dest)

    return _copy_file


##################################
## Begin rest of PyDoit tasks here
##################################


def task_config():
    """Create empty directories for data and output if they don't exist"""
    return {
        "actions": ["python ./src/settings.py"],
        "targets": [DATA_DIR, OUTPUT_DIR],
        "file_dep": ["./src/settings.py"],
        "clean": [],
    }


def task_pull():
    """Pull raw data from WRDS."""
    yield {
        "name": "crsp_daily",
        "doc": "CRSP daily stock file and the CRSP-TAQ symbol link",
        "actions": ["python ./src/pull_crsp_daily.py"],
        "targets": [
            DATA_DIR / "crsp_daily.parquet",
            DATA_DIR / "taq_crsp_link.parquet",
        ],
        "file_dep": ["./src/settings.py", "./src/pull_crsp_daily.py"],
        "clean": True,
    }


def task_universe():
    """Apply the paper's sample screens to pick the stocks in each day."""
    return {
        "actions": ["python ./src/build_universe.py"],
        "targets": [
            DATA_DIR / "universe.parquet",
            DATA_DIR / "universe_attrition.parquet",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/build_universe.py",
            DATA_DIR / "crsp_daily.parquet",
            DATA_DIR / "taq_crsp_link.parquet",
        ],
        "clean": True,
    }


def task_minute_panel():
    """Build the one-minute return panel, one task per trading day.

    One task per day so that a failed or added day rebuilds on its own instead
    of re-pulling the whole sample, and so the same tasks can be farmed out to
    a cluster job array.
    """
    for date in PANEL_DATES:
        stamp = date.strftime("%Y%m%d")
        yield {
            "name": stamp,
            "actions": [f"python ./src/run_minute_panel.py {date:%Y-%m-%d}"],
            "targets": [
                DATA_DIR / "minute_panel" / f"returns_{stamp}.parquet",
                DATA_DIR / "minute_panel" / f"coverage_{stamp}.parquet",
            ],
            "file_dep": [
                "./src/settings.py",
                "./src/build_minute_panel.py",
                "./src/run_minute_panel.py",
                "./src/pull_taq_minute_bars.py",
                DATA_DIR / "universe.parquet",
            ],
            "clean": True,
        }


def task_panel_exhibits():
    """Summary tables and the coverage figure for the minute panel."""
    coverage_files = [
        DATA_DIR / "minute_panel" / f"coverage_{d:%Y%m%d}.parquet" for d in PANEL_DATES
    ]
    yield {
        "name": "summary_table",
        "actions": ["python ./src/table_panel_summary.py"],
        "targets": [
            OUTPUT_DIR / "table_panel_summary.tex",
            OUTPUT_DIR / "table_universe_attrition.tex",
        ],
        "file_dep": [
            "./src/table_panel_summary.py",
            DATA_DIR / "universe_attrition.parquet",
            *coverage_files,
        ],
        "clean": True,
    }
    yield {
        "name": "staleness_sensitivity",
        "doc": "Measures what the stale-price cutoff costs the panel",
        "actions": ["python ./src/table_staleness_sensitivity.py"],
        "targets": [OUTPUT_DIR / "table_staleness_sensitivity.tex"],
        "file_dep": [
            "./src/table_staleness_sensitivity.py",
            "./src/build_minute_panel.py",
            DATA_DIR / "universe.parquet",
        ],
        "clean": True,
    }
    yield {
        "name": "coverage_figure",
        "actions": ["python ./src/figure_coverage.py"],
        "targets": [OUTPUT_DIR / "figure_coverage.png"],
        "file_dep": ["./src/figure_coverage.py", *coverage_files],
        "clean": True,
    }


notebook_tasks = {}


# fmt: off
def task_run_notebooks():
    """Preps the notebooks for presentation format.
    Execute notebooks if the script version of it has been changed.
    """
    for notebook in notebook_tasks:
        pyfile_path = Path(notebook_tasks[notebook]["path"])
        notebook_path = pyfile_path.with_suffix("")  # strips .py, leaves .ipynb
        notebook_name = notebook_path.stem  # e.g. "01_example_notebook_interactive"
        yield {
            "name": notebook,
            "actions": [
                """python -c "import sys; from datetime import datetime; print(f'Start """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
                f"jupytext --to notebook --output {notebook_path} {pyfile_path}",
                jupyter_execute_notebook(notebook_path),
                jupyter_to_html(notebook_path),
                mv(notebook_path, OUTPUT_DIR),
                """python -c "import sys; from datetime import datetime; print(f'End """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
            ],
            "file_dep": [
                pyfile_path,
                *notebook_tasks[notebook]["file_dep"],
            ],
            "targets": [
                OUTPUT_DIR / f"{notebook_name}.html",
                *notebook_tasks[notebook]["targets"],
            ],
            "clean": True,
        }
# fmt: on

###############################################################
## Task below is for LaTeX compilation
###############################################################


def task_compile_latex_docs():
    """Compile the LaTeX documents to PDFs"""
    file_dep = [
        "./reports/report_example.tex",
        "./reports/my_article_header.sty",
        "./reports/slides_example.tex",
        "./reports/my_beamer_header.sty",
        "./reports/my_common_header.sty",
        "./reports/report_simple_example.tex",
        "./reports/slides_simple_example.tex",
    ]
    targets = [
        "./reports/report_example.pdf",
        "./reports/slides_example.pdf",
        "./reports/report_simple_example.pdf",
        "./reports/slides_simple_example.pdf",
    ]

    return {
        "actions": [
            # My custom LaTeX templates
            "latexmk -xelatex -halt-on-error -cd ./reports/report_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/report_example.tex",  # Clean
            "latexmk -xelatex -halt-on-error -cd ./reports/slides_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/slides_example.tex",  # Clean
            # Simple templates based on small adjustments to Overleaf templates
            "latexmk -xelatex -halt-on-error -cd ./reports/report_simple_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/report_simple_example.tex",  # Clean
            "latexmk -xelatex -halt-on-error -cd ./reports/slides_simple_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/slides_simple_example.tex",  # Clean
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }


sphinx_targets = [
    "./docs/index.html",
]


def task_build_chartbook_site():
    """Compile Sphinx Docs"""
    notebook_scripts = [
        Path(notebook_tasks[notebook]["path"]) for notebook in notebook_tasks
    ]
    file_dep = [
        "./README.md",
        "./chartbook.toml",
        *notebook_scripts,
    ]

    return {
        "actions": [
            "chartbook docs build -f",
        ],  # Use docs as build destination
        "targets": sphinx_targets,
        "file_dep": file_dep,
        "task_dep": [
            "run_notebooks",
        ],
        "clean": True,
    }


def task_run_pytest():
    """Run pytest and save results to OUTPUT_DIR"""
    src_py_files = list(Path("./src").glob("*.py"))
    test_output = OUTPUT_DIR / "pytest_results.xml"

    def run_pytest():
        import subprocess

        result = subprocess.run(
            ["pytest", f"--junitxml={test_output}"],
            check=False,  # returncode is handled below
        )
        if result.returncode != 0:
            # Remove the XML so doit won't consider the target up-to-date
            Path(test_output).unlink(missing_ok=True)
            raise RuntimeError(f"pytest failed with exit code {result.returncode}")

    return {
        "actions": [run_pytest],
        "targets": [test_output],
        "file_dep": src_py_files,
        "clean": True,
        "verbosity": 2,
    }
