# Flight Ranking Competition - Quick Start

## Current Status
✅ **Baseline model working** - ready to submit to Kaggle  
🎯 **Goal**: Predict which flight business travelers will choose  
💰 **Prize**: $10,000 ($5,000 for 1st place if >70% accuracy)  
⏰ **Time left**: 8 days

## Files You Need

### Core Scripts
- `fix_submission.py` - Fix and submit your baseline model
- `complete_submission.py` - Generate complete submissions 
- `foolproof_processor.py` - Process large datasets reliably

### Analysis & Models
- `flight_ranking_notebook.ipynb` - Jupyter notebook for data exploration
- `requirements.txt` - Python packages needed

### Key Folders
- `data/` - Competition data (train.parquet, test.parquet)
- `submissions/` - Generated submission files
- `kaggle_env/` - Python virtual environment
- `archive/` - Old scripts (can ignore)

## Next Steps

1. **Submit baseline** (if not done):
   ```bash
   python fix_submission.py
   # Upload: submissions/validated_submission.parquet
   ```

2. **Build better model** using your airline expertise:
   - Morning departure preferences
   - Corporate policy compliance  
   - Frequent flyer benefits
   - Route efficiency patterns

3. **Improve iteratively** based on leaderboard feedback

## Your Competitive Advantage

As an airline employee, you understand business traveler behavior better than most competitors:
- **Time preferences**: Early departures for day trips
- **Corporate rules**: Policy compliance vs convenience trade-offs
- **Loyalty programs**: Status benefits influence choices significantly
- **Route knowledge**: Hub strategies, connection preferences

## Competition Details

- **Problem**: Learning-to-rank (predict top choice from flight options)
- **Metric**: HitRate@3 (correct flight in top 3 predictions)  
- **Data**: 18M training examples, 127 features per flight
- **Scoring**: Only groups with >10 options count toward final score

Good luck! 🛫