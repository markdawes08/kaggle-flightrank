# FlightRank 2025: Business Traveler Recommendation System

A machine learning solution for predicting flight choices in the Aeroclub RecSys Cup competition.

## Overview

This repository contains a ranking model that predicts which flight options business travelers will select from search results. The model uses flight attributes, pricing data, and traveler preferences to rank options within each search session.

**Competition**: [FlightRank 2025: Aeroclub RecSys Cup](https://www.kaggle.com/c/aeroclub-recsys-2025)  
**Evaluation Metric**: HitRate@3 (fraction of sessions where the correct flight appears in top 3 predictions)

## Project Structure

```
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── flight_ranking_notebook.ipynb      # Data exploration and analysis
├── foolproof_processor.py             # Robust data processing pipeline
├── data/                              # Competition datasets (not tracked)
├── submissions/                       # Model predictions (not tracked)
└── models/                           # Trained models (not tracked)
```

## Setup

1. **Environment**:
   ```bash
   python -m venv kaggle_env
   source kaggle_env/bin/activate
   pip install -r requirements.txt
   ```

2. **Data**: Download competition data to `data/` folder:
   - `train.parquet` (18M rows, 127 features)
   - `test.parquet` (6.9M rows)
   - `sample_submission.parquet`

## Approach

### Data Processing
- Chunk-based processing for memory efficiency
- Feature engineering focused on business traveler behavior
- Robust handling of missing values and edge cases

### Model Features
- **Price ranking**: Position within search results
- **Temporal patterns**: Departure time preferences
- **Corporate compliance**: Travel policy adherence
- **Loyalty factors**: Frequent flyer considerations

### Baseline Model
Simple price-based ranking achieves ~25-30% HitRate@3, providing a foundation for more sophisticated approaches.

## Key Features

The model analyzes business travel patterns through:
- Temporal booking preferences
- Corporate travel policy compliance
- Route optimization factors  
- Frequent flyer program utilization

## Competition Performance

- **Target**: HitRate@3 ≥ 0.70 for bonus prize eligibility
- **Current**: Baseline implementation ready for submission
- **Approach**: Iterative improvement based on leaderboard feedback

## Development

The codebase prioritizes reliability and memory efficiency when working with large datasets (500MB+ parquet files). All processing scripts include fallback methods and error handling for robust execution across different system configurations.