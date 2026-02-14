from setuptools import setup, find_packages
import pathlib

# Read README safely
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="Essentiax",
    version="0.3.0",  # ⬆️ Major version bump for Feature Engineering module
    author="Shubham Wagh",
    author_email="waghshubham197@gmail.com",
    description="A next-generation Python library for smart EDA, cleaning, and interpretability in ML.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShubhamWagh/EssentiaX",
    
    packages=find_packages(),
    include_package_data=True,

    # ⬇️ CRITICAL UPDATE: Added visualization and analysis libraries + Feature Engineering
    install_requires=[
        "pandas>=1.0",
        "numpy>=1.20",
        "matplotlib>=3.0",
        "seaborn>=0.11",
        "scikit-learn>=1.0",
        "rich>=10.0",      # For beautiful console UI
        "openpyxl>=3.0",   # For Excel files
        "plotly>=5.0",     # Interactive visualizations
        "scipy>=1.7",      # Statistical analysis
        "kaleido>=0.2.1"   # Static image export for Plotly
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