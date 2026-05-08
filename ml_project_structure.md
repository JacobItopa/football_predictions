# Machine Learning Project Directory Structure

The best practice for structuring a Machine Learning project is heavily inspired by the widely adopted **Cookiecutter Data Science** project template. This structure ensures that your project is logical, easily reproducible, and scalable.

## Recommended Directory Structure

```text
my_ml_project/
├── data/
│   ├── raw/           # The original, immutable raw data dumps. NEVER edit these.
│   ├── interim/       # Intermediate data that has been partially transformed.
│   ├── processed/     # The final, clean datasets ready for modeling.
│   └── external/      # Any external data from third-party sources or APIs.
│
├── models/            # Trained and serialized models (.pkl, .joblib, .h5, etc).
│
├── notebooks/         # Jupyter notebooks used for EDA and initial experiments.
│                      # Use a naming convention like: 01-eda-initial.ipynb
│
├── references/        # Data dictionaries, manuals, papers, or API documentation.
│
├── reports/           # Generated analysis artifacts (HTML, PDF, logs).
│   └── figures/       # Generated graphics and charts to be used in reporting.
│
├── src/               # The actual production-ready source code.
│   ├── __init__.py    # Makes src a Python module.
│   ├── config.py      # Configuration variables (paths, credentials, hyperparams).
│   ├── data/          # Scripts to load, process, or fetch data.
│   ├── features/      # Scripts that build features from raw/interim data.
│   ├── models/        # Scripts to train, evaluate, and predict with your models.
│   └── visualization/ # Scripts to create plots and evaluate metrics.
│
├── .gitignore         # Files to be ignored by Git (e.g., raw data, secrets, .env).
├── README.md          # The top-level README describing the project and how to run it.
├── requirements.txt   # Or pyproject.toml / environment.yml. Project dependencies.
└── .env               # Environment variables (API keys, DB secrets). NEVER commit this.
```

### Why this structure works best:
1. **Data is Separate from Code:** You ensure your raw data is never accidentally mutated, and large data files can be easily excluded from Git version control.
2. **Notebooks are for Exploration, `src/` is for Production:** Jupyter notebooks get messy fast. When a feature or model idea works in a notebook, the best practice is to move it into clean, reproducible `.py` scripts inside the `src/` folder.
3. **Reproducibility:** Anyone pulling down the code can install `requirements.txt`, run the scripts in `src/data` and `src/features`, and get the exact same results.
