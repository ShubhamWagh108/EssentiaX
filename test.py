# test.py
"""
Test Script for Essentia Smart EDA
-----------------------------------
This script loads a dataset and runs the smart_eda() function
to perform automated exploratory data analysis.
"""

import pandas as pd
from essentiax.smartEDA import smart_eda

# 🧠 Load dataset (example: Netflix dataset)
df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\essentia\netflix_titles.csv.csv")

# ✅ Run Smart EDA
smart_eda(df, sample_size=1000)
