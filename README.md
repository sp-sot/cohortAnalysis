# Cohort Payment Data Analysis Package

This Python package provides tools for analyzing a company's cohort payment data, enabling deeper insights into customer behavior, revenue patterns, and cohort trends. It is designed to be modular and extensible, making it suitable for various types of analyses, from exploratory to in-depth statistical modeling.

## Features
- **Preprocessing Tools**: Prepare raw payment data for analysis by cleaning, transforming, and structuring it into analyzable formats.
- **Core Analysis**: Built-in functions for cohort-based analysis, such as churn calculation, retention modeling, and lifetime value estimation.
- **Ad Hoc Analysis**: Flexible tools for custom analyses and insights tailored to specific business needs.
- **Statistical Modeling**: Integration with lifelines and statsmodels for advanced survival analysis and regression modeling.

## Installation

### Requirements
This package is developed using Python 3.6.5 and depends on the libraries specified in `requirements.txt`.

### Setup
1. Create a virtual environment:
    ```bash
    python3.6.5 -m venv .env
    source .env/bin/activate
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## File Structure
The package includes the following files:
- **`analyzer.py`**: Contains the main analysis functions, including cohort comparisons, churn analysis, and revenue metrics.
- **`preprocessor.py`**: Handles data cleaning and preprocessing tasks to prepare raw data for analysis.
- **`ad_hoc_analysis.py`**: A module for performing custom analyses based on user-defined needs.
- **`test_analyser.py`**: A test suite built with `pytest` to validate the functionality of the core analysis and preprocessing modules.
- **`underwriting_template.ipynb`**: A Jupyter Notebook showcasing how to use the package for a typical analysis workflow.
- **`README.md`**: This document.
- **`requirements.txt`**: Lists all Python dependencies required for the package.

## Usage
### Notebook Example
Check `underwriting_template.ipynb` for a step-by-step guide on using the package.

## Testing
Run the test suite to ensure everything is working correctly:
```bash
pytest test_analyser.py --cov=.
