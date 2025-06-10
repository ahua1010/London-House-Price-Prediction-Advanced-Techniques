import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from feature_engineering import engineer_features
import joblib
from datetime import datetime
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import os
import pickle
from sklearn.feature_selection import SelectFromModel
import json


def get_model_dict():
    """
    返回一個包含所有基礎模型的字典
    每個模型都是已經初始化的實例
    """
    return {
        'etr': ExtraTreesRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        'xgb': xgb.XGBRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            objective='reg:squarederror'
        ),
        'lgb': lgb.LGBMRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            objective='regression'
        ),
        'cat': CatBoostRegressor(
            iterations=100,
            random_state=42,
            verbose=0,
            loss_function='RMSE',
            eval_metric='RMSE'
        ),
        'lr': LinearRegression(n_jobs=-1)
    }

def get_most_important_features(X_train, y_train, n_features, model_name='lgb'):
    """
    Select most important features using specified model.
    
    Args:
        X_train: Training features
        y_train: Target variable
        n_features: Number of features to select
        model_name: Model to use for feature selection ('lgb', 'xgb', 'etr')
        
    Returns:
        List of selected feature names
    """
    if model_name == 'lgb':
        model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    elif model_name == 'xgb':
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    else:  # default to ExtraTrees
        model = ExtraTreesRegressor(n_estimators=100, random_state=42)
    
    # Fit model and get feature importance
    model.fit(X_train, y_train)
    
    if model_name == 'lgb':
        importances = model.feature_importances_
    elif model_name == 'xgb':
        importances = model.get_booster().get_score(importance_type='gain')
        importances = [importances.get(f, 0) for f in X_train.columns]
    else:
        importances = model.feature_importances_
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Select top n_features
    selected_features = feature_importance['feature'].head(n_features).tolist()
    print(f"\n=== 特徵選擇過程 ===")
    print(f"原始特徵數量: {len(X_train.columns)}")
    print(f"原始特徵列表: {X_train.columns.tolist()}")
    print(f"\n{model_name.upper()} 特徵重要性:")
    print(feature_importance)
    print(f"\n選中的特徵數量: {len(selected_features)}")
    print(f"選中的特徵列表: {selected_features}")
    
    return selected_features

# 主訓練循環
def run_time_series_stacking(
    df: pd.DataFrame,
    target_col: str,
    n_splits: int = 5,
    fe_config_base: dict = None,
    quick_test: bool = False,
    top_k_features: int = 30
) -> dict:
    """
    Run time series cross-validation with model stacking.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        n_splits: Number of time series splits
        fe_config_base: Base feature engineering configuration
        quick_test: Whether to run a quick test with fewer features
        top_k_features: Number of top features to select
        
    Returns:
        Dictionary containing final configuration
    """
    # Initialize model dictionary
    models = get_model_dict()
    
    # Initialize storage for predictions
    oof_predictions = pd.DataFrame(index=df.index)
    test_predictions = pd.DataFrame(index=df.index)
    
    # Initialize storage for selected features and final models
    selected_features = None
    final_models = {}
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df), 1):
        print(f"\n=== 開始第 {fold} 折訓練 ===")
        
        # Split data
        train_data = df.iloc[train_idx]
        val_data = df.iloc[val_idx]
        
        # Feature engineering - now handles price transformation internally
        train_processed = engineer_features(train_data.copy(), config=fe_config_base, is_train=True)
        val_processed = engineer_features(val_data.copy(), config=fe_config_base, is_train=False)

        # Separate features and target
        y_train = train_processed[target_col]
        X_train = train_processed.drop(columns=[target_col])
        
        y_val = np.log1p(val_data[target_col]) # We need to manually transform the validation target
        X_val = val_processed
        
        # Feature selection (only in first fold)
        if fold == 1:
            # Ensure target is not in selected features
            features_for_selection = [col for col in X_train.columns if col != target_col]
            selected_features = get_most_important_features(
                X_train[features_for_selection],
                y_train,
                top_k_features,
                model_name='lgb'
            )
        
        # Train base models
        for name, model in models.items():
            print(f"訓練模型: {name}")
            try:
                model.fit(X_train[selected_features], y_train)
                # Predict on validation set
                oof_pred = model.predict(X_val[selected_features])
                oof_predictions.loc[val_idx, name] = oof_pred
                
                if fold == n_splits:
                    final_models[name] = model
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        # Calculate fold OOF MAE on original scale for better interpretation
        fold_oof_preds_mean = oof_predictions.loc[val_idx, list(models.keys())].mean(axis=1)
        original_scale_preds = np.expm1(fold_oof_preds_mean)
        original_scale_true = val_data[target_col] # Use original non-transformed price

        fold_oof_mae = mean_absolute_error(original_scale_true, original_scale_preds)
        print(f"Fold {fold} OOF MAE (on original price scale): {fold_oof_mae:,.2f}")
    
    print("\n訓練元模型...")

    # The target for meta-model is the log-transformed price from the full dataset
    full_y = np.log1p(df[target_col])
    
    # Train meta-model
    meta_features = oof_predictions[list(models.keys())].fillna(oof_predictions.mean())
    meta_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
    meta_model.fit(meta_features, full_y)
    
    # Calculate final OOF MAE on original scale
    final_oof_predictions = oof_predictions[list(final_models.keys())].fillna(oof_predictions.mean()).mean(axis=1)
    final_original_scale_preds = np.expm1(final_oof_predictions)
    final_original_scale_true = df[target_col]

    final_oof_mae = mean_absolute_error(final_original_scale_true, final_original_scale_preds)
    print(f"Final OOF MAE (on original price scale): {final_oof_mae:,.2f}")
    
    # Save final model and configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"output/models/stacking_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save individual models
    for name, model in final_models.items():
        model_path = os.path.join(save_dir, f"{name}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    # Save meta model
    meta_model_path = os.path.join(save_dir, "meta_model.pkl")
    with open(meta_model_path, 'wb') as f:
        pickle.dump(meta_model, f)
    
    # Create and save complete configuration
    final_config = {
        'selected_features': selected_features,
        'model_paths': {
            name: os.path.join(save_dir, f"{name}_model.pkl")
            for name in final_models.keys()
        },
        'meta_model_path': meta_model_path,
        'oof_mae': final_oof_mae,
        'timestamp': timestamp,
        'feature_engineering_config': fe_config_base,
        'models': final_models,
        'meta_model': meta_model
    }
    
    # Save complete configuration as a single pickle file
    config_path = os.path.join(save_dir, "config.pkl")
    with open(config_path, 'wb') as f:
        pickle.dump(final_config, f)
    
    return final_config

if __name__ == "__main__":
    # Set paths
    train_path = "train.csv"
    test_path = "test.csv"
    output_path = "output"
    
    try:
        # Read data
        print("=== 讀取數據 ===")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Run model with quick test mode
        print("\n=== 開始訓練 ===")
        final_config = run_time_series_stacking(
            df=train_df,
            target_col='price',
            n_splits=5,
            fe_config_base={},
            quick_test=True,
            top_k_features=30
        )
        print(f"\n訓練完成。模型與設定已儲存於 {final_config['meta_model_path']}")

    except Exception as e:
        print("\n=== 訓練過程中發生錯誤 ===")
        print(f"錯誤類型: {type(e).__name__}")
        print(f"錯誤信息: {e}\n")
        import traceback
        print("完整錯誤堆疊:")
        traceback.print_exc() 