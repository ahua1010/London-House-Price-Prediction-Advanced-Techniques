import pandas as pd
import numpy as np
import joblib
import os
from feature_engineering import engineer_features
from datetime import datetime
import pickle

def load_latest_model(output_path="output"):
    """
    載入最新的模型和配置文件
    """
    # 獲取 models 目錄下最新的運行文件夾
    models_dir = os.path.join(output_path, 'models')
    run_folders = [f for f in os.listdir(models_dir) if f.startswith('stacking_')]
    if not run_folders:
        raise FileNotFoundError("No model runs found in output/models directory")
    
    latest_run = sorted(run_folders)[-1]
    run_path = os.path.join(models_dir, latest_run)
    
    # 載入配置文件
    config_path = os.path.join(run_path, "config.pkl")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    try:
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        raise
    
    return config

def predict(test_path, output_path):
    # 加載配置
    config = load_latest_model(output_path)
    if config is None:
        raise ValueError("無法加載配置")
    
    # 讀取測試數據
    print("=== 讀取測試數據 ===")
    test_df = pd.read_csv(test_path)
    print(f"測試數據列名: {test_df.columns.tolist()}")
    
    # 檢查並處理 ID 列
    if 'ID' not in test_df.columns:
        print("警告: 測試數據中沒有 'ID' 列，將使用索引作為 ID")
        test_df['ID'] = test_df.index
    else:
        print(f"找到 'ID' 列，數據類型: {test_df['ID'].dtype}")
        # 確保 ID 是整數類型
        test_df['ID'] = test_df['ID'].astype(int)
    
    # 保存原始數據的副本，用於最終的 ID 對應
    original_test_df = test_df.copy()
    
    # 特徵工程（不包含 ID）
    print("=== 特徵工程 ===")
    test_features = engineer_features(test_df.copy(), config=config['feature_engineering_config'], is_train=False)
    
    # 確保使用與訓練時相同的特徵
    selected_features = config.get('selected_features', [])
    if not selected_features:
        raise ValueError("配置文件中缺少選定的特徵列表")
    
    # 檢查並對齊特徵
    available_features = [col for col in selected_features if col in test_features.columns]
    missing_features = set(selected_features) - set(available_features)
    if missing_features:
        print(f"警告: 以下特徵在測試集中缺失: {missing_features}")
        # 為缺失的特徵添加零值列，而不是使用可能洩漏數據的全局均值
        for feature in missing_features:
            test_features[feature] = 0
    
    # 選擇特徵
    X_test = test_features[available_features].copy()
    print(f"使用 {len(available_features)} 個特徵進行預測")
    
    # 使用基礎模型進行預測
    base_predictions = pd.DataFrame()
    for name, model in config['models'].items():
        try:
            pred = model.predict(X_test)
            base_predictions[name] = pred
        except Exception as e:
            print(f"警告: 模型 {name} 預測失敗 - {e}")
            base_predictions[name] = 0
    
    # 使用元模型進行最終預測
    meta_features = base_predictions.fillna(base_predictions.mean())
    log_scale_predictions = config['meta_model'].predict(meta_features)

    # 將預測結果從 log 尺度反轉換為原始價格尺度
    final_predictions = np.expm1(log_scale_predictions)
    
    # 創建提交文件
    submission = pd.DataFrame({
        'ID': original_test_df['ID'],
        'price': final_predictions
    })
    
    # 保存預測結果
    timestamp = datetime.now().strftime('%m%d_%H%M')
    output_dir = os.path.join(output_path, 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'submission_{timestamp}.csv')
    submission.to_csv(output_file, index=False)
    print(f"預測結果已保存至: {output_file}")
    
    return submission

if __name__ == "__main__":
    test_path = "test.csv"
    output_path = "output"
    predict(test_path, output_path) 