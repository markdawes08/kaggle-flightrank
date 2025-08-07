#!/usr/bin/env python3
"""
Advanced ranking model for flight selection
Target: Move from 0.20 to 0.40+ HitRate@3
"""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import gc
import json
from datetime import datetime

print("=== BUILDING ADVANCED RANKING MODEL ===")
print(f"Started at: {datetime.now()}")

# Configuration
SAMPLE_TRAINING = True  # Set False for full training (will take longer)
TRAIN_SAMPLES = 500000 if SAMPLE_TRAINING else None
N_FOLDS = 3

def load_data_efficiently():
    """Load data with only useful columns"""
    print("\n1. Loading data efficiently...")
    
    # Define columns we actually need
    essential_cols = [
        'Id', 'ranker_id', 'selected',
        'totalPrice', 'taxes',
        'frequentFlyer', 'isVip', 'bySelf',
        'corporateTariffCode', 'searchRoute',
    ]
    
    # Time-based columns (we'll parse these)
    time_cols = [col for col in ['legs0_departureAt', 'legs0_arrivalAt', 
                                  'legs1_departureAt', 'legs1_arrivalAt',
                                  'requestDate']]
    
    # Duration columns
    duration_cols = ['legs0_duration', 'legs1_duration']
    
    # Policy columns
    policy_cols = ['pricingInfo_isAccessTP', 'pricingInfo_passengerCount']
    
    # Airline columns
    airline_cols = [col for col in ['legs0_segments0_marketingCarrier_code',
                                    'legs0_segments0_operatingCarrier_code',
                                    'legs0_segments0_cabinClass',
                                    'legs0_segments0_baggageAllowance_quantity']]
    
    # Flight rules
    rules_cols = ['miniRules0_statusInfos', 'miniRules1_statusInfos',
                  'miniRules0_monetaryAmount', 'miniRules1_monetaryAmount']
    
    # Load train data
    train_file = pq.ParquetFile('data/train.parquet')
    all_columns = set(train_file.schema_arrow.names)
    
    # Only request columns that exist
    load_cols = essential_cols.copy()
    for col_list in [time_cols, duration_cols, policy_cols, airline_cols, rules_cols]:
        load_cols.extend([col for col in col_list if col in all_columns])
    
    print(f"Loading {len(load_cols)} columns...")
    
    if SAMPLE_TRAINING:
        # Load sample for faster iteration
        train = train_file.read(columns=load_cols).to_pandas()[:TRAIN_SAMPLES]
    else:
        train = train_file.read(columns=load_cols).to_pandas()
    
    print(f"Loaded train: {len(train):,} rows")
    
    # Load test similarly
    test_file = pq.ParquetFile('data/test.parquet')
    test_cols = [col for col in load_cols if col != 'selected']
    test = test_file.read(columns=test_cols).to_pandas()
    
    print(f"Loaded test: {len(test):,} rows")
    
    return train, test

def engineer_features(df, is_train=True):
    """Create powerful features for ranking"""
    print("\n2. Engineering features...")
    
    # Price features
    print("   - Price features")
    df['price_per_tax'] = df['totalPrice'] / (df['taxes'] + 1)
    df['tax_ratio'] = df['taxes'] / df['totalPrice']
    
    # Rank within group
    df['price_rank'] = df.groupby('ranker_id')['totalPrice'].rank()
    df['price_pct'] = df.groupby('ranker_id')['totalPrice'].rank(pct=True)
    df['is_cheapest'] = (df['price_rank'] == 1).astype(int)
    df['is_most_expensive'] = (df['price_rank'] == df.groupby('ranker_id')['price_rank'].transform('max')).astype(int)
    
    # Group statistics
    df['group_size'] = df.groupby('ranker_id')['Id'].transform('count')
    df['price_vs_mean'] = df['totalPrice'] / df.groupby('ranker_id')['totalPrice'].transform('mean')
    df['price_vs_median'] = df['totalPrice'] / df.groupby('ranker_id')['totalPrice'].transform('median')
    df['price_std_in_group'] = df.groupby('ranker_id')['totalPrice'].transform('std')
    
    # Time features (if available)
    if 'legs0_departureAt' in df.columns:
        print("   - Time features")
        try:
            df['departure_dt'] = pd.to_datetime(df['legs0_departureAt'])
            df['departure_hour'] = df['departure_dt'].dt.hour
            df['departure_day'] = df['departure_dt'].dt.dayofweek
            
            # Business hours
            df['is_morning'] = ((df['departure_hour'] >= 6) & (df['departure_hour'] <= 10)).astype(int)
            df['is_evening'] = ((df['departure_hour'] >= 17) & (df['departure_hour'] <= 20)).astype(int)
            df['is_business_hours'] = ((df['departure_hour'] >= 8) & (df['departure_hour'] <= 18)).astype(int)
            
            # Rank departure times
            df['departure_rank'] = df.groupby('ranker_id')['departure_hour'].rank()
        except:
            print("     Warning: Could not parse departure times")
    
    # Duration features
    if 'legs0_duration' in df.columns:
        print("   - Duration features")
        try:
            # Convert duration to minutes
            df['duration_minutes'] = pd.to_timedelta(df['legs0_duration']).dt.total_seconds() / 60
            df['duration_rank'] = df.groupby('ranker_id')['duration_minutes'].rank()
            df['is_shortest'] = (df['duration_rank'] == 1).astype(int)
            
            # Duration vs price trade-off
            df['minutes_per_dollar'] = df['duration_minutes'] / df['totalPrice']
        except:
            print("     Warning: Could not parse durations")
    
    # Corporate policy
    if 'pricingInfo_isAccessTP' in df.columns:
        print("   - Policy features")
        df['is_policy_compliant'] = df['pricingInfo_isAccessTP'].fillna(0).astype(int)
        df['policy_rank'] = df.groupby('ranker_id')['is_policy_compliant'].rank(ascending=False)
    
    # Frequent flyer
    if 'frequentFlyer' in df.columns:
        df['has_frequent_flyer'] = df['frequentFlyer'].notna().astype(int)
    
    # VIP status
    if 'isVip' in df.columns:
        df['is_vip'] = df['isVip'].fillna(0).astype(int)
    
    # Cabin class
    if 'legs0_segments0_cabinClass' in df.columns:
        print("   - Cabin features")
        df['is_business_class'] = (df['legs0_segments0_cabinClass'] == 2.0).astype(int)
        df['is_premium'] = (df['legs0_segments0_cabinClass'] == 4.0).astype(int)
    
    # Flexibility features (cancellation/exchange rules)
    if 'miniRules0_statusInfos' in df.columns:
        print("   - Flexibility features")
        df['can_cancel'] = (df['miniRules0_statusInfos'] != 0).astype(int)
        df['can_exchange'] = (df['miniRules1_statusInfos'] != 0).astype(int)
        df['is_flexible'] = ((df['can_cancel'] == 1) | (df['can_exchange'] == 1)).astype(int)
    
    # Route complexity (single vs round trip)
    if 'searchRoute' in df.columns:
        df['is_round_trip'] = df['searchRoute'].str.contains('/').fillna(False).astype(int)
    
    print(f"   Created {len([col for col in df.columns if col not in ['Id', 'ranker_id', 'selected']])} features")
    
    return df

def train_lightgbm_ranker(train_df, feature_cols):
    """Train LightGBM ranking model with cross-validation"""
    print("\n3. Training LightGBM ranking model...")
    
    # Prepare data
    X = train_df[feature_cols]
    y = train_df['selected']
    groups = train_df['ranker_id']
    
    # Get group sizes for LightGBM
    group_sizes = train_df.groupby('ranker_id').size().values
    
    # Parameters for ranking
    lgb_params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [3],
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
        'num_threads': -1
    }
    
    # Cross-validation
    gkf = GroupKFold(n_splits=N_FOLDS)
    cv_scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n   Fold {fold + 1}/{N_FOLDS}")
        
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Get group sizes for train and val
        train_groups = train_df.iloc[train_idx].groupby('ranker_id').size().values
        val_groups = train_df.iloc[val_idx].groupby('ranker_id').size().values
        
        # Create LightGBM datasets
        train_set = lgb.Dataset(X_train, label=y_train, group=train_groups)
        val_set = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=train_set)
        
        # Train model
        model = lgb.train(
            lgb_params,
            train_set,
            valid_sets=[val_set],
            num_boost_round=300,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )
        
        models.append(model)
        
        # Validate with HitRate@3
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        val_df = pd.DataFrame({
            'ranker_id': train_df.iloc[val_idx]['ranker_id'],
            'pred': val_pred,
            'actual': y_val
        })
        
        # Calculate HitRate@3
        hit_count = 0
        total_groups = 0
        
        for ranker_id, group in val_df.groupby('ranker_id'):
            if len(group) <= 10:  # Skip small groups
                continue
            
            # Get top 3 predictions
            top3 = group.nlargest(3, 'pred')
            if top3['actual'].sum() > 0:
                hit_count += 1
            total_groups += 1
        
        hitrate = hit_count / total_groups if total_groups > 0 else 0
        cv_scores.append(hitrate)
        print(f"   Fold {fold + 1} HitRate@3: {hitrate:.4f}")
    
    print(f"\n   Average CV HitRate@3: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame()
    for i, model in enumerate(models):
        fold_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importance(importance_type='gain'),
            'fold': i
        })
        feature_importance = pd.concat([feature_importance, fold_importance])
    
    # Average importance across folds
    mean_importance = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
    
    print("\n   Top 10 most important features:")
    for feat, imp in mean_importance.head(10).items():
        print(f"   - {feat}: {imp:.1f}")
    
    return models, np.mean(cv_scores)

def create_submission(test_df, models, feature_cols):
    """Create submission using ensemble of models"""
    print("\n4. Creating submission...")
    
    # Predict with each model and average
    predictions = np.zeros(len(test_df))
    
    for i, model in enumerate(models):
        print(f"   Predicting with model {i+1}/{len(models)}")
        pred = model.predict(test_df[feature_cols], num_iteration=model.best_iteration)
        predictions += pred / len(models)
    
    # Create submission
    submission = test_df[['Id', 'ranker_id']].copy()
    submission['pred'] = predictions
    
    # Rank within each group (higher prediction = better = rank 1)
    submission['selected'] = submission.groupby('ranker_id')['pred'].rank(
        method='first', ascending=False
    ).astype(int)
    
    # Save
    submission[['Id', 'ranker_id', 'selected']].to_parquet(
        'submissions/lgb_ranking_submission.parquet', index=False
    )
    
    print(f"   Submission saved: submissions/lgb_ranking_submission.parquet")
    
    return submission

def main():
    """Main pipeline"""
    
    # Load data
    train, test = load_data_efficiently()
    
    # Engineer features
    train = engineer_features(train, is_train=True)
    test = engineer_features(test, is_train=False)
    
    # Select feature columns (exclude identifiers and target)
    exclude_cols = ['Id', 'ranker_id', 'selected', 'departure_dt']
    feature_cols = [col for col in train.columns if col not in exclude_cols and col in test.columns]
    
    print(f"\nUsing {len(feature_cols)} features for training")
    
    # Train models
    models, cv_score = train_lightgbm_ranker(train, feature_cols)
    
    # Create submission
    submission = create_submission(test, models, feature_cols)
    
    # Save results summary
    results = {
        'cv_hitrate3': float(cv_score),
        'expected_improvement': float(cv_score / 0.20044),  # vs your baseline
        'n_features': len(feature_cols),
        'n_models': len(models),
        'feature_cols': feature_cols,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n=== COMPLETE ===")
    print(f"Expected HitRate@3: {cv_score:.4f} (vs baseline: 0.20044)")
    print(f"Expected improvement: {cv_score/0.20044:.1f}x")
    print(f"Expected leaderboard position: ~250-300 (rough estimate)")
    print("\nUpload: submissions/lgb_ranking_submission.parquet")
    
    return cv_score

if __name__ == "__main__":
    score = main()
    print(f"\nFinished at: {datetime.now()}")