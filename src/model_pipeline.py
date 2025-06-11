import os
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from numba import cuda

# 匯入模型和評估工具
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# --- 全局配置 ---
# 透過 numba 檢查 GPU 是否可用，避免引入 torch 的重依賴
try:
    cuda.select_device(0)
    USE_GPU = True
except:
    USE_GPU = False
QUICK_TEST = False # 設定為 True 以快速測試流程，減少折數和特徵組合

# 匯入特徵工程模組
from feature_engineering import engineer_features, select_features

def train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, models, meta_model, y_train_orig):
    """
    訓練基模型和元模型，並返回預測結果。
    """
    oof_preds = pd.DataFrame(index=X_val.index)
    test_preds = pd.DataFrame(index=X_test.index)
    
    for name, model in models.items():
        print(f"訓練模型: {name}")
        try:
            model.fit(X_train, y_train)
            
            # --- ADDED: In-fold MAE calculation ---
            train_preds_log = model.predict(X_train)
            train_preds_orig = np.expm1(train_preds_log)
            in_fold_mae = mean_absolute_error(y_train_orig, train_preds_orig)
            print(f"    - 折內 MAE: {in_fold_mae:,.2f}")
            # --- END ADDED ---

            oof_preds[name] = model.predict(X_val)
            test_preds[name] = model.predict(X_test)
        except Exception as e:
            print(f"訓練模型 {name} 時發生錯誤: {e}")
            oof_preds[name] = np.nan
            test_preds[name] = np.nan
            
    # 訓練元模型
    print("\n訓練元模型...")
    meta_model.fit(oof_preds.fillna(oof_preds.mean()), y_val)
    
    # 元模型進行最終預測
    final_oof_pred = meta_model.predict(oof_preds)
    final_test_pred = meta_model.predict(test_preds)
    
    return pd.DataFrame(final_oof_pred, index=X_val.index, columns=['meta_pred']), \
           final_test_pred, \
           models

def run_time_series_stacking(df, test_df, target_col, models, meta_model, n_splits=5, random_state=42, top_k=40, use_gpu=False, quick_test=False):
    """
    執行時間序列交叉驗證的 Stacking。
    (已重構以防止數據洩漏)
    """
    print("\n=== 開始訓練 ===")
    
    # --- 修正數據洩漏: 按時間排序數據 ---
    if 'sale_year' in df.columns and 'sale_month' in df.columns:
        print("-> 正在按時間排序數據以進行時間序列交叉驗證...")
        df = df.sort_values(['sale_year', 'sale_month']).reset_index()

    X = df.drop(columns=[target_col], errors='ignore')
    y = df[target_col]

    # --- 解決 'ID' 鍵錯誤 ---
    # 將 'ID' 從特徵中移除，並保存以供後續使用
    if 'ID' in X.columns:
        original_ids = X['ID']
        X = X.drop(columns=['ID'])
    else:
        # 如果 'ID' 已經是索引，直接使用
        original_ids = X.index
    
    # --- 修正數據洩漏: 使用 TimeSeriesSplit ---
    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    kf = TimeSeriesSplit(n_splits=n_splits)
    
    oof_predictions = np.zeros(len(X))
    test_predictions_sum = np.zeros(len(test_df))
    oof_mae_scores = []
    
    fold_feature_sets = []

    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        print(f"\n=== 開始第 {fold + 1}/{n_splits} 折訓練 ===")
        X_train_fold_raw, X_val_fold_raw = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        
        print(f"本折原始維度: Train={X_train_fold_raw.shape}, Val={X_val_fold_raw.shape}")

        # --- 特徵工程 (在 Fold 內部進行) ---
        X_train_with_target = X_train_fold_raw.copy()
        X_train_with_target[target_col] = y_train_fold

        config_fold = {}
        X_train_featured, y_train_log_processed, config_fold = engineer_features(
            X_train_with_target, is_train=True, config=config_fold, quick_test=quick_test
        )
        
        X_val_featured, _, _ = engineer_features(
            X_val_fold_raw.copy(), is_train=False, config=config_fold, quick_test=quick_test
        )
        test_df_featured_fold, _, _ = engineer_features(
            test_df.copy(), is_train=False, config=config_fold, quick_test=quick_test
        )

        # --- 特徵選擇 (在 Fold 內部進行) ---
        selected_features, config_fold = select_features(
            X_train_featured, y_train_log_processed, config=config_fold, k=top_k, use_gpu=use_gpu
        )
        fold_feature_sets.append(set(selected_features))
        
        # 確保所有選中的特徵都存在於所有數據集中
        final_selected_features = [f for f in selected_features if f in X_train_featured.columns and f in X_val_featured.columns and f in test_df_featured_fold.columns]
        
        X_train_selected = X_train_featured[final_selected_features]
        X_val_selected = X_val_featured[final_selected_features]
        test_selected = test_df_featured_fold[final_selected_features]
        
        print(f"特徵工程與選擇後，維度: Train={X_train_selected.shape}, Val={X_val_selected.shape}, Test={test_selected.shape}")

        # 模型訓練與預測
        y_val_log = np.log1p(y_val_fold)
        fold_oof_preds, fold_test_preds, _ = train_and_evaluate_models(
            X_train_selected, y_train_log_processed, 
            X_val_selected, y_val_log,
            test_selected, 
            models, meta_model,
            y_train_fold # Pass original scale target for in-fold MAE
        )
        
        oof_predictions[val_index] = fold_oof_preds['meta_pred'].values
        test_predictions_sum += fold_test_preds
        
        original_scale_preds = np.expm1(fold_oof_preds['meta_pred'].values)
        fold_oof_mae = mean_absolute_error(y_val_fold, original_scale_preds)
        oof_mae_scores.append(fold_oof_mae)
        print(f"Fold {fold + 1} OOF MAE (on original price scale): {fold_oof_mae:,.2f}")

    final_oof_mae = np.mean(oof_mae_scores)
    print(f"\nFinal OOF MAE (on original price scale): {final_oof_mae:,.2f}")
    
    # 使用保存的原始 ID 創建 Series，確保索引正確
    oof_preds_series = pd.Series(oof_predictions, index=original_ids)
    final_test_preds = test_predictions_sum / n_splits
    
    # --- 訓練最終模型 (在所有數據上) ---
    print("\n=== 訓練最終模型 (使用所有數據) ===")
    final_config = {}
    
    X_full_with_target = df.copy()
    X_full_featured, y_full_log, final_config = engineer_features(
        X_full_with_target, is_train=True, config=final_config, quick_test=quick_test
    )

    if fold_feature_sets:
        final_features = list(set.intersection(*fold_feature_sets))
        if not final_features:
            print("警告：各折之間沒有共同的特徵，將使用所有折特徵的聯集。")
            final_features = list(set.union(*fold_feature_sets))
        print(f"最終模型將使用 {len(final_features)} 個特徵。")
    else:
        final_features = X_full_featured.columns.tolist()
        print("警告: 未找到交叉驗證的特徵集，最終模型將使用所有特徵。")
    
    # 確保最終特徵存在於完全工程化的數據框中
    final_features = [f for f in final_features if f in X_full_featured.columns]
    final_config['selected_features'] = final_features
    X_full_selected = X_full_featured[final_features]

    final_trained_models = {}
    for name, model in models.items():
        print(f"訓練最終模型: {name}")
        model.fit(X_full_selected, y_full_log)
        final_trained_models[name] = model
        
    final_config['trained_models'] = final_trained_models

    return oof_preds_series, final_test_preds, final_config

def save_artifacts(config, oof_preds, test_preds, submission_df):
    """
    儲存所有訓練產物。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"output/models/stacking_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # 儲存模型
    trained_models = config.get('trained_models', {})
    if trained_models:
        for name, model in trained_models.items():
            joblib.dump(model, os.path.join(save_dir, f"{name}.pkl"))

    # 創建一個乾淨的 config 來儲存，避免循環引用
    config_to_save = {
        'selected_features': config.get('selected_features'),
        'better_features_list': config.get('better_features_list'),
        # 如果有其他需要保存的 scaler 或 imputer，可以在這裡添加
    }
    
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        # 使用自訂的轉換器處理 numpy 類型
        json.dump(config_to_save, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    # 儲存 OOF 預測
    oof_df = pd.DataFrame({'ID': oof_preds.index, 'oof_predictions': oof_preds})
    oof_df.to_csv(os.path.join(save_dir, 'oof_predictions.csv'), index=False)

    # 儲存測試集預測並創建提交檔案
    submission_df['price'] = np.expm1(test_preds)
    submission_df.to_csv(os.path.join(save_dir, 'submission.csv'), index=False)
    
    print(f"\n訓練完成。所有產物已儲存於 {save_dir}")
    return save_dir

def main():
    # 讀取數據
    print("=== 讀取數據 ===")
    try:
        train_df = pd.read_csv('train.csv', index_col='ID')
        test_df = pd.read_csv('test.csv', index_col='ID')
        submission_df = pd.read_csv('sample_submission.csv')
    except FileNotFoundError as e:
        print(f"錯誤: 找不到數據文件 - {e}")
        return

    n_splits = 2 if QUICK_TEST else 5
    print(f"\n--- 快速測試模式: {'啟用' if QUICK_TEST else '禁用'} (使用 {n_splits} 折) ---")

    if QUICK_TEST:
        print("--- 快速測試模式: 正在縮減初始特徵集 ---")
        # 定義快速測試所需的核心原始特徵
        core_features = [
            'floorAreaSqM',
            'latitude', 'longitude',
            'propertyType',
            'sale_year',
            'postcode',
        ]
        # 確保目標欄位 'price' 被保留在訓練集中，並處理數據集中可能不存在的欄位
        train_cols_to_keep = [col for col in core_features if col in train_df.columns] + ['price']
        test_cols_to_keep = [col for col in core_features if col in test_df.columns]
        
        train_df = train_df[list(set(train_cols_to_keep))]
        test_df = test_df[list(set(test_cols_to_keep))]
        print(f"初始特徵已縮減為 {len(test_cols_to_keep)} 個核心特徵。")

    # 設置 GPU 加速
    print(f"\n--- GPU 加速已{'啟用' if USE_GPU else '禁用'} ---")

    # 定義模型
    models = {
        'etr': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1, bootstrap=True),
        'xgb': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
                                max_depth=7, min_child_weight=1, gamma=0.1, subsample=0.8,
                                colsample_bytree=0.8, reg_alpha=0.005, random_state=42),
        'lgb': lgb.LGBMRegressor(random_state=42),
        'cat': cb.CatBoostRegressor(random_state=42, verbose=0),
    }
    meta_model = Ridge()

    if USE_GPU:
        print("\n--- GPU 加速已啟用 ---")
        # 更新模型以使用 GPU
        models['xgb'] = xgb.XGBRegressor(random_state=42, tree_method='gpu_hist')
        models['lgb'] = lgb.LGBMRegressor(random_state=42, device='gpu')
        models['cat'] = cb.CatBoostRegressor(random_state=42, verbose=0, task_type='GPU')
        # rf 和 meta_model (LinearRegression) 不支援 GPU，保持原樣
    else:
        print("\n--- GPU 加速已禁用，使用 CPU ---")

    # 執行訓練
    oof_preds, test_preds, final_config = run_time_series_stacking(
        train_df, test_df, 'price', models, meta_model, 
        n_splits=n_splits, use_gpu=USE_GPU, quick_test=QUICK_TEST
    )
    
    # 儲存結果
    save_artifacts(final_config, oof_preds, test_preds, submission_df)


if __name__ == "__main__":
    main() 