# Replication Study: Word Embeddings Quantify 100 Years of Gender and Ethnic Stereotypes

This repository contains a replication and extension of Garg et al.'s (2018) seminal work on quantifying historical stereotypes using word embeddings. This project was completed as part of QTM/DATASCI 340: Text as Data at Emory University.

## Original Paper

**Garg, N., Schiebinger, L., Jurafsky, D. & Zou, J.** (2018). Word embeddings quantify 100 years of gender and ethnic stereotypes. *Proceedings of the National Academy of Sciences*, 115(16), E3635-E3644. [doi:10.1073/pnas.1720347115](https://doi.org/10.1073/pnas.1720347115)

- Original repository: [github.com/nikhgarg/EmbeddingDynamicStereotypes](https://github.com/nikhgarg/EmbeddingDynamicStereotypes)
- PDF: [Available here](http://gargnikhil.com/files/pdfs/GSJZ18_embedstereotypes.pdf)

## Team

- **Shuyang Yu**
- **Eika Zhou**
- **Hanwen Zhang**
- **Jeffrey Zheng**


Instructors: Prof. Sandeep Soni, Prof. Steph Buongiorno

## Project Overview

This project aims to:

1. **Replicate** the original study's methodology and key findings, which used word2vec embeddings to measure changes in gender and ethnic stereotypes across the 20th century in American English text
2. **Extend** the analysis through cross-linguistic comparison, examining stereotype patterns in Chinese language corpora
3. **Validate** the robustness of the original findings by reproducing correlations between embedding-based bias measures and external benchmarks

### Key Findings from Original Study

The original paper demonstrated that:
- Word embeddings reliably capture semantic biases and stereotypes
- These biases correlate with historical Census data and psychological surveys
- Occupational gender bias decreased from 1910s to 1990s but remained substantial
- Ethnic stereotypes show measurable changes aligned with historical events

## Repository Structure

```
.
├── changes_over_time_sgns_google.py
├── create_final_plots_all_sgns_google_robust.py
├── latexify.py
├── plot_creation.py
├── run_params_sgns_google.csv
├── utilities.py
│
├── data/
│   └── vocab_counts/
│
├── dataset_utilities/
│   ├── coha_vector_clean.py
│   ├── coha_vector_normalize.py
│   └── pipeline.py
│
├── plots/
│   ├── appendix/
│   │   ├── ethnicity/
│   │   └── gender/
│   ├── ethnicity/
│   └── gender/
│
├── regressions/
│
└── run_results/
    └── finalrun.csv
```

## Data Sources

### Word Embeddings

This replication uses publicly available pre-trained embeddings:

1. **COHA (Corpus of Historical American English)** - 1910s-1990s
   - Source: [Stanford HistWords Project](https://nlp.stanford.edu/projects/histwords/)
   - Type: SGNS (Skip-gram with Negative Sampling)
   - Used for temporal analysis of American English

2. **Google News word2vec** - Contemporary baseline
   - Source: [Google word2vec](https://code.google.com/archive/p/word2vec/)
   - Trained on ~100 billion words from Google News

3. **Additional datasets** (for extensions):
   - Chinese HistWords embeddings

### External Benchmarks

- **U.S. Census Occupational Data** (1910-1990): Gender and ethnic percentages by occupation
- **Psychological Stereotype Surveys**: Princeton stereotype studies, Williams & Best adjective ratings, MTurk surveys

## Replication Results

### Output Files

The analysis generates three types of outputs:

1. **Plots** (`plots/` directory):
   - `gender/`: Gender bias visualizations including correlations with Census data
   - `ethnicity/`: Ethnic stereotype visualizations and temporal trends
   - `appendix/`: Supplementary analyses (frequency, variance, robustness checks)

2. **Regressions** (`regressions/` directory):
   - CSV files containing regression coefficients, p-values, and R² statistics
   - Correlation results between embedding biases and external benchmarks

3. **Run Results** (`run_results/` directory):
   - Logs of analysis runs with parameters and vectors

## Extensions

### Cross-Linguistic Analysis (Primary Extension)

We extend the original work by comparing stereotype patterns across languages:

**Research Questions:**
- Do similar stereotypes exist in Chinese language embeddings?
- How do cultural differences manifest in embedding-based bias measures?
- Are temporal trends in stereotypes consistent across languages?

**Methodology:**
- Apply same bias measurement framework to Chinese corpora
- Compare gender and ethnic bias patterns between English and Chinese
- Analyze culture-specific stereotypes

**Status:** [In progress]

---

**Course**: QTM/DATASCI 340: Text as Data  
**Semester**: Fall 2025
