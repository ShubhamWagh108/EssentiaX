from setuptools import setup, find_packages
import pathlib

# Read README safely
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="Essentiax",
    version="1.0.2",  # Bug fix: Add SmartClean class wrapper
    author="Shubham Wagh",
    author_email="waghshubham197@gmail.com",
    description="Complete ML automation platform with AutoML, Feature Engineering, AI Insights, and Interactive Dashboards.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShubhamWagh108/EssentiaX",
    
    packages=find_packages(),
    include_package_data=True,

    install_requires=[
        "pandas>=1.0",
        "numpy>=1.20",
        "matplotlib>=3.0",
        "seaborn>=0.11",
        "scikit-learn>=1.0",
        "rich>=10.0",
        "openpyxl>=3.0",
        "plotly>=5.0",
        "scipy>=1.7",
        "kaleido>=0.2.1",
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        "catboost>=1.0.0",
        "optuna>=3.0.0",
        "shap>=0.41.0",
        "imbalanced-learn>=0.9.0",
        "joblib>=1.1.0"
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
    ],

    python_requires=">=3.7",
)