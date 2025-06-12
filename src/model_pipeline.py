import os
import pandas as pd
import numpy as np
import joblib
import json
from numba import cuda
import logging
from datetime import datetime

# 匯入模型和評估工具
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.base import clone
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# 匯入新的時間序列元模型
from time_series_meta_models import (
    TimeSeriesAwareMetaModel,
    HierarchicalTimeSeriesMetaModel,
    TimeSeriesCrossValidationMetaModel,
    EnsembleTimeSeriesMetaModel
)

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

def train_base_models(X_train, y_train, X_val, y_val, X_test, models):
    """
    訓練基模型並返回預測結果
    """
    print("\n=== 基模型訓練開始 ===")
    print(f"訓練集形狀: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"驗證集形狀: X_val={X_val.shape}, y_val={y_val.shape}")
    print(f"測試集形狀: X_test={X_test.shape}")
    # print(f"訓練集特徵: {X_train.columns.tolist()}")
    print(f"訓練集目標變量範圍: [{y_train.min()}, {y_train.max()}]")
    
    # 初始化預測結果
    oof_preds = {}
    test_preds = {}
    
    # 訓練每個模型
    for name, model in models.items():
        print(f"\n訓練模型: {name}")
        print(f"模型參數: {model.get_params()}")
        
        try:
            # 訓練模型
            model.fit(X_train, y_train)
            print("模型訓練完成，進行預測...")
            
            # 進行預測
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            
            # 確保預測結果是數值類型
            oof_preds[name] = pd.Series(val_pred, index=X_val.index, dtype=float)
            test_preds[name] = pd.Series(test_pred, index=X_test.index, dtype=float)
            
            print(f"預測完成，驗證集預測範圍: [{val_pred.min()}, {val_pred.max()}]")
            
        except Exception as e:
            print(f"模型 {name} 訓練失敗: {str(e)}")
            # 如果模型訓練失敗，使用 NaN 填充預測結果
            oof_preds[name] = pd.Series(np.nan, index=X_val.index, dtype=float)
            test_preds[name] = pd.Series(np.nan, index=X_test.index, dtype=float)
    
    # 檢查訓練成功的模型數量
    successful_models = sum(1 for pred in oof_preds.values() if not pred.isna().all())
    print(f"\n=== 基模型訓練完成 ===")
    print(f"成功訓練的模型數量: {successful_models}")
    
    # 將預測結果轉換為 DataFrame
    oof_preds_df = pd.DataFrame(oof_preds)
    test_preds_df = pd.DataFrame(test_preds)
    
    print(f"預測結果形狀: oof_preds={oof_preds_df.shape}, test_preds={test_preds_df.shape}")
    
    return oof_preds, test_preds, successful_models

def check_data_quality(X, y, stage="training"):
    """檢查數據質量"""
    print(f"\n=== {stage} 數據品質檢查 ===")
    
    # 檢查數值型特徵
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_X = X[numeric_cols]
        print(f"數值型特徵範圍: [{numeric_X.min().min()}, {numeric_X.max().max()}]")
        print(f"數值型特徵數量: {len(numeric_cols)}")
    
    # 檢查類別型特徵
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"類別型特徵數量: {len(categorical_cols)}")
        for col in categorical_cols:
            unique_count = X[col].nunique()
            print(f"- {col}: {unique_count} 個唯一值")
    
    # 檢查目標變量
    print(f"\n目標變量檢查:")
    print(f"y 範圍: [{y.min()}, {y.max()}]")
    print(f"y 均值: {y.mean()}")
    print(f"y 標準差: {y.std()}")
    
    # 檢查缺失值
    print(f"\n缺失值檢查:")
    print(f"X 缺失值總數: {X.isna().sum().sum()}")
    print(f"y 缺失值數量: {y.isna().sum()}")
    
    # 檢查數據形狀
    print(f"\n數據形狀:")
    print(f"X 形狀: {X.shape}")
    print(f"y 形狀: {y.shape}")

def check_predictions(preds, y_true, stage="validation"):
    """檢查預測值"""
    print(f"\n=== {stage} 預測值檢查 ===")
    print(f"預測值範圍: [{preds.min()}, {preds.max()}]")
    print(f"真實值範圍: [{y_true.min()}, {y_true.max()}]")
    print(f"預測值均值: {preds.mean()}")
    print(f"真實值均值: {y_true.mean()}")
    print(f"預測值標準差: {preds.std()}")
    print(f"真實值標準差: {y_true.std()}")

def evaluate_model(y_true, y_pred, stage="validation"):
    """評估模型並提供詳細的檢查信息"""
    print(f"\n=== {stage} 評估 ===")
    # 檢查原始尺度
    print("原始尺度:")
    print(f"y_true 範圍: [{y_true.min()}, {y_true.max()}]")
    print(f"y_pred 範圍: [{y_pred.min()}, {y_pred.max()}]")
    
    # 轉換到對數尺度
    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(y_pred)
    print("\n對數尺度:")
    print(f"y_true_log 範圍: [{y_true_log.min()}, {y_true_log.max()}]")
    print(f"y_pred_log 範圍: [{y_pred_log.min()}, {y_pred_log.max()}]")
    
    # 計算 MAE
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\nMAE: {mae:.4f}")
    return mae

def run_time_series_stacking(df, test_df, target_col, models, meta_model, n_splits=5, random_state=42, top_k=40, use_gpu=False, quick_test=False):
    """
    執行時間序列交叉驗證的 Stacking，使用不同的時間序列元模型
    """
    print("\n=== 開始訓練 ===")
    
    # --- 修正數據洩漏: 按時間排序數據 ---
    if 'sale_year' in df.columns and 'sale_month' in df.columns:
        print("-> 正在按時間排序數據以進行時間序列交叉驗證...")
        df = df.sort_values(['sale_year', 'sale_month']).reset_index(drop=True)
        print(f"-> 排序後數據的 index 範圍: [{df.index.min()}, {df.index.max()}]")
        print(f"-> 排序後數據的 index 是否連續: {df.index.is_monotonic_increasing and df.index.is_unique}")

    X = df.drop(columns=[target_col], errors='ignore')
    y = df[target_col]

    if quick_test:
        # 檢查原始數據
        check_data_quality(X, y, "原始數據")

    # 對目標變量進行 log1p 轉換
    y_log = np.log1p(y)
    print("\n=== 目標變量轉換檢查 ===")
    print(f"原始 y 範圍: [{y.min()}, {y.max()}]")
    print(f"轉換後 y_log 範圍: [{y_log.min()}, {y_log.max()}]")
    
    # --- 解決 'ID' 鍵錯誤 ---
    if 'ID' in X.columns:
        original_ids = X['ID']
        X = X.drop(columns=['ID'])
        print(f"-> 已移除 ID 欄位，原始 ID 數量: {len(original_ids)}")
    else:
        original_ids = X.index
        print(f"-> 使用 index 作為 ID，數量: {len(original_ids)}")
    
    # --- 修正數據洩漏: 使用 TimeSeriesSplit ---
    kf = TimeSeriesSplit(n_splits=n_splits)
    
    # 初始化預測結果
    oof_predictions = pd.DataFrame(index=range(len(X)), columns=models.keys(), dtype=float)
    test_predictions = pd.DataFrame(index=test_df.index, columns=models.keys(), dtype=float)
    test_predictions = test_predictions.fillna(0)  # 初始化為 0
    
    print(f"\n-> OOF predictions 初始化形狀: {oof_predictions.shape}")
    print(f"-> Test predictions 初始化形狀: {test_predictions.shape}")
    
    # 時間特徵
    time_features = ['sale_year']
    
    # 用於存儲每個折的基模型評估結果
    fold_metrics = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(X, y_log)):
        print(f"\n=== 開始第 {fold + 1}/{n_splits} 折訓練 ===")
        print(f"-> 訓練集索引範圍: [{train_index.min()}, {train_index.max()}], 大小: {len(train_index)}")
        print(f"-> 驗證集索引範圍: [{val_index.min()}, {val_index.max()}], 大小: {len(val_index)}")
        
        X_train_fold_raw, X_val_fold_raw = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y_log.iloc[train_index], y_log.iloc[val_index]
        
        if quick_test:
            # 檢查折的數據
            check_data_quality(X_train_fold_raw, y_train_fold, f"折 {fold + 1} 訓練集")
            check_data_quality(X_val_fold_raw, y_val_fold, f"折 {fold + 1} 驗證集")
        
        # 特徵工程
        X_train_with_target = X_train_fold_raw.copy()
        X_train_with_target[target_col] = y_train_fold  # 使用已經 log1p 轉換過的 y_train_fold

        config_fold = {}
        print("\n=====訓練集特徵處理開始=====")
        X_train_featured, _, config_fold = engineer_features(  # 移除 y_train_log_processed
            X_train_with_target, is_train=True, config=config_fold, quick_test=quick_test
        )
        print("\n=====驗證集特徵處理開始=====")        
        X_val_featured, _, _ = engineer_features(
            X_val_fold_raw.copy(), is_train=False, config=config_fold, quick_test=quick_test
        )
        print("\n=====測試集特徵處理開始=====")
        test_df_featured_fold, _, _ = engineer_features(
            test_df.copy(), is_train=False, config=config_fold, quick_test=quick_test
        )

        if quick_test:
            # 檢查特徵工程後的數據
            check_data_quality(X_train_featured, y_train_fold, f"折 {fold + 1} 特徵工程後訓練集")  # 使用 y_train_fold
            check_data_quality(X_val_featured, y_val_fold, f"折 {fold + 1} 特徵工程後驗證集")

        # 特徵選擇
        selected_features, config_fold = select_features(
            X_train_featured, y_train_fold, config=config_fold, k=top_k, use_gpu=USE_GPU
        )
        
        # 確保所有選中的特徵都存在於所有數據集中
        final_selected_features = [f for f in selected_features if f in X_train_featured.columns and f in X_val_featured.columns and f in test_df_featured_fold.columns]
        print(f"\n-> 最終選擇的特徵數量: {len(final_selected_features)}")
        
        X_train_selected = X_train_featured[final_selected_features]
        X_val_selected = X_val_featured[final_selected_features]
        test_selected = test_df_featured_fold[final_selected_features]
        
        if quick_test:
            # 檢查特徵選擇後的數據
            check_data_quality(X_train_selected, y_train_fold, f"折 {fold + 1} 特徵選擇後訓練集")
            check_data_quality(X_val_selected, y_val_fold, f"折 {fold + 1} 特徵選擇後驗證集")
        
        # 只訓練基模型
        fold_oof_preds, fold_test_preds, _ = train_base_models(
            X_train_selected, y_train_fold, 
            X_val_selected, y_val_fold,
            test_selected, 
            models
        )
        
        # 收集預測結果
        print("\n-> 正在收集預測結果...")
        for model_name in models.keys():
            # 確保預測結果是數值類型，並且正確對齊索引
            oof_predictions.loc[val_index, model_name] = fold_oof_preds[model_name].values
            test_predictions[model_name] += fold_test_preds[model_name].values / n_splits
            
        # 計算當前折的基模型評估指標
        fold_metrics_fold = {}
        for model_name in models.keys():
            val_pred = fold_oof_preds[model_name]
            # 轉換回原始尺度進行評估
            val_pred_original = np.expm1(val_pred)
            y_val_original = np.expm1(y_val_fold)
            
            # 檢查預測值
            check_predictions(val_pred_original, y_val_original, f"折 {fold + 1} {model_name}")
            
            # 使用評估函數
            mae = evaluate_model(y_val_original, val_pred_original, f"折 {fold + 1} {model_name}")
            fold_metrics_fold[model_name] = mae
            print(f"-> {model_name} 在折 {fold + 1} 的 MAE: {mae:.4f}")
        
        fold_metrics.append(fold_metrics_fold)
        
        # 檢查當前折的預測結果
        print(f"-> 當前折 OOF predictions 非空值數量: {oof_predictions.loc[val_index].notna().sum().sum()}")
        print(f"-> 當前折 OOF predictions 形狀: {oof_predictions.loc[val_index].shape}")
    
    # 計算並打印所有基模型的平均 MAE
    print("\n=== 基模型評估結果 ===")
    for model_name in models.keys():
        avg_mae = np.mean([fold_metrics[fold][model_name] for fold in range(n_splits)])
        print(f"-> {model_name} 平均 MAE: {avg_mae:.4f}")
    
    # 確保所有預測結果都是數值類型
    oof_predictions = oof_predictions.astype(float)
    test_predictions = test_predictions.astype(float)
    
    # 檢查是否有 NaN 值
    print("\n檢查預測結果中的 NaN 值:")
    print("OOF predictions NaN count:", oof_predictions.isna().sum().sum())
    print("Test predictions NaN count:", test_predictions.isna().sum().sum())
    
    # 檢查 OOF predictions 的覆蓋率
    total_predictions = oof_predictions.shape[0] * oof_predictions.shape[1]
    non_nan_predictions = oof_predictions.notna().sum().sum()
    coverage_rate = non_nan_predictions / total_predictions * 100
    print(f"\n-> OOF predictions 覆蓋率: {coverage_rate:.2f}%")
    
    # 只保留有預測值的 row
    valid_idx = ~oof_predictions.isna().any(axis=1)
    oof_predictions_valid = oof_predictions[valid_idx]
    y_valid = y_log[valid_idx]
    
    print(f"\n-> 有效預測數量: {len(oof_predictions_valid)}")
    print(f"-> 有效預測比例: {len(oof_predictions_valid)/len(oof_predictions)*100:.2f}%")
    
    # 檢查預測值的範圍
    print("\n檢查預測值範圍:")
    for col in oof_predictions_valid.columns:
        print(f"{col} 預測範圍: [{oof_predictions_valid[col].min():.4f}, {oof_predictions_valid[col].max():.4f}]")
    
    # 訓練四種不同的時間序列元模型
    print("\n=== 訓練時間序列元模型 ===")
    
    # 1. 時間序列感知元模型
    print("\n訓練時間序列感知元模型...")
    ts_aware_meta = TimeSeriesAwareMetaModel(
        clone(meta_model),
        time_features=time_features
    )
    # 合併預測結果和時間特徵
    oof_with_time_valid = pd.concat([oof_predictions_valid, X.loc[valid_idx, time_features]], axis=1)
    test_with_time = pd.concat([test_predictions, test_df[time_features]], axis=1)
    ts_aware_meta.fit(oof_with_time_valid, y_valid)
    ts_aware_preds = ts_aware_meta.predict(test_with_time)
    
    # 2. 分層時間序列元模型
    print("\n訓練分層時間序列元模型...")
    hierarchical_meta = HierarchicalTimeSeriesMetaModel(
        clone(meta_model)
    )
    hierarchical_meta.fit(oof_predictions_valid, y_valid)
    hierarchical_preds = hierarchical_meta.predict(test_predictions)
    
    # 3. 時間序列交叉驗證元模型
    print("\n訓練時間序列交叉驗證元模型...")
    ts_cv_meta = TimeSeriesCrossValidationMetaModel(
        clone(meta_model)
    )
    ts_cv_meta.fit(oof_predictions_valid, y_valid)
    ts_cv_preds = ts_cv_meta.predict(test_predictions)
    
    # 4. 集成時間序列元模型
    print("\n訓練集成時間序列元模型...")
    ensemble_meta = EnsembleTimeSeriesMetaModel(
        clone(meta_model),
        time_features=time_features
    )
    ensemble_meta.fit(oof_with_time_valid, y_valid)
    ensemble_preds = ensemble_meta.predict(test_with_time)
    
    # 計算並打印元模型的評估指標
    print("\n=== 元模型評估結果 ===")
    meta_metrics = {}
    
    # 計算每個元模型的訓練集預測
    ts_aware_preds_train = ts_aware_meta.predict(oof_with_time_valid)
    hierarchical_preds_train = hierarchical_meta.predict(oof_predictions_valid)
    ts_cv_preds_train = ts_cv_meta.predict(oof_predictions_valid)
    ensemble_preds_train = ensemble_meta.predict(oof_with_time_valid)
    
    # 轉換回原始尺度進行評估
    y_valid_original = np.expm1(y_valid)
    
    # 計算每個元模型的 MAE
    meta_metrics['ts_aware'] = mean_absolute_error(y_valid_original, np.expm1(ts_aware_preds_train))
    meta_metrics['hierarchical'] = mean_absolute_error(y_valid_original, np.expm1(hierarchical_preds_train))
    meta_metrics['ts_cv'] = mean_absolute_error(y_valid_original, np.expm1(ts_cv_preds_train))
    meta_metrics['ensemble'] = mean_absolute_error(y_valid_original, np.expm1(ensemble_preds_train))
    
    print("\n元模型 MAE:")
    for model_name, mae in meta_metrics.items():
        print(f"-> {model_name}: {mae:.4f}")
    
    # 保存預測結果
    predictions = {
        'ts_aware': np.expm1(ts_aware_preds),
        'hierarchical': np.expm1(hierarchical_preds),
        'ts_cv': np.expm1(ts_cv_preds),
        'ensemble': np.expm1(ensemble_preds)
    }
    
    # 保存配置
    final_config = {
        'models': models,
        'meta_models': {
            'ts_aware': ts_aware_meta,
            'hierarchical': hierarchical_meta,
            'ts_cv': ts_cv_meta,
            'ensemble': ensemble_meta
        },
        'mae': meta_metrics['ensemble'],  # 使用集成模型的 MAE
        'base_model_metrics': {model_name: np.mean([fold_metrics[fold][model_name] for fold in range(n_splits)]) 
                             for model_name in models.keys()},
        'meta_model_metrics': meta_metrics
    }
    
    return predictions, final_config

def save_artifacts(config, predictions, submission_df):
    """
    保存模型和預測結果到以時間戳記和 MAE 命名的資料夾中
    """
    from datetime import datetime
    import shutil
    
    # 從配置中獲取 MAE
    mae = config.get('mae', 'unknown')
    mae_str = f"MAE_{mae:.4f}" if isinstance(mae, float) else "MAE_unknown"
    
    # 創建時間戳記
    timestamp = datetime.now().strftime("%m%d_%H%M")
    
    # 創建輸出資料夾
    output_dir = f"output/{timestamp}_{mae_str}"
    
    try:
        # 創建新的輸出目錄
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存所有物件
        print(f"\n正在保存模型和預測結果到 {output_dir}...")
        
        # 保存配置
        joblib.dump(config, f'{output_dir}/config.pkl')
        
        # 保存預測結果
        joblib.dump(predictions, f'{output_dir}/predictions.pkl')
        
        # 保存提交文件
        joblib.dump(submission_df, f'{output_dir}/submission.pkl')
        
        # 分別保存每個模型
        if 'models' in config:
            for name, model in config['models'].items():
                joblib.dump(model, f'{output_dir}/base_model_{name}.pkl')
        if 'meta_models' in config:
            for name, meta_model in config['meta_models'].items():
                joblib.dump(meta_model, f'{output_dir}/meta_model_{name}.pkl')
        
        # 保存每個提交檔
        for name, preds in predictions.items():
            df = submission_df.copy()
            df['prediction'] = preds
            df.to_csv(f'{output_dir}/submission_{name}.csv', index=False)
        
        print(f"\n所有檔案已成功保存到: {output_dir}")
        
    except OSError as e:
        print(f"錯誤：保存文件時發生問題 - {e}")
        print("嘗試清理更多空間...")
        # 清理所有舊的輸出
        for old_dir in os.listdir('output'):
            if os.path.isdir(os.path.join('output', old_dir)):
                shutil.rmtree(os.path.join('output', old_dir))
        # 重新創建輸出目錄
        os.makedirs(output_dir, exist_ok=True)
        # 只保存最重要的文件
        joblib.dump(config, f'{output_dir}/config.pkl')
        for name, preds in predictions.items():
            df = submission_df.copy()
            df['prediction'] = preds
            df.to_csv(f'{output_dir}/submission_{name}.csv', index=False)
        print(f"\n已保存最小必要文件到: {output_dir}")
    except Exception as e:
        print(f"錯誤：保存過程中發生未預期的錯誤 - {e}")
        raise

def main():
    # 設置日誌
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = f'{log_dir}/model_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    try:
        # 讀取數據
        logging.info("=== 讀取數據 ===")
        try:
            df = pd.read_csv('data/train.csv')
            test_df = pd.read_csv('data/test.csv')
            logging.info(f"成功讀取數據：訓練集 {df.shape}，測試集 {test_df.shape}")
        except FileNotFoundError as e:
            logging.error(f"錯誤: 找不到數據文件 - {e}")
            return
        except Exception as e:
            logging.error(f"讀取數據時發生錯誤: {e}")
            return

        # 快速測試模式：減少特徵數量
        if QUICK_TEST:
            logging.info("\n=== 快速測試模式：減少特徵數量 ===")
            # 定義核心特徵
            core_features = [
                'ID',
                'floorAreaSqM',
                'latitude', 'longitude',
                'propertyType',
                'sale_year',
                'postcode',
                'price'  # 目標變量
            ]
            # 確保所有核心特徵都存在
            train_cols = [col for col in core_features if col in df.columns]
            test_cols = [col for col in core_features if col in test_df.columns and col != 'price']
            
            logging.info(f"原始特徵數量: {df.shape[1]}")
            df = df[train_cols]
            test_df = test_df[test_cols]
            logging.info(f"快速測試模式特徵數量: {df.shape[1]}")
        
        # 定義模型
        models = {
            'lgb': lgb.LGBMRegressor(
                random_state=42,
                n_estimators=100 if QUICK_TEST else 1000,
                learning_rate=0.1,
                num_leaves=31,
                verbose=-1
            ),
            'xgb': xgb.XGBRegressor(
                random_state=42,
                n_estimators=100 if QUICK_TEST else 1000,
                learning_rate=0.1,
                max_depth=6,
                verbosity=0
            ),
            'cat': cb.CatBoostRegressor(
                random_state=42,
                iterations=100 if QUICK_TEST else 1000,
                learning_rate=0.1,
                depth=6,
                verbose=False
            ),
            'lr': LinearRegression(
                n_jobs=-1  # 使用所有可用的 CPU 核心
            ),
            'ridge': Ridge(
                alpha=1.0,  # 正則化強度
                random_state=42,
                solver='auto'  # 自動選擇最佳求解器
            )
        }
        
        # 定義元模型
        meta_model = ExtraTreesRegressor(
            random_state=42,
            n_estimators=100 if QUICK_TEST else 500,
            max_depth=10
        )
        
        # 執行訓練
        logging.info("開始執行模型訓練...")
        predictions, config = run_time_series_stacking(
            df, test_df, 'price',
            models, meta_model,
            n_splits=2 if QUICK_TEST else 5,  # 快速測試時減少折數
            top_k=20 if QUICK_TEST else 40,   # 快速測試時減少特徵數量
            use_gpu=USE_GPU,
            quick_test=QUICK_TEST
        )
        
        # 保存結果
        submission_df = pd.DataFrame({'ID': test_df['ID']})
        save_artifacts(config, predictions, submission_df)
        
        logging.info("模型訓練和保存完成！")
        
    except Exception as e:
        logging.error(f"執行過程中發生錯誤: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 