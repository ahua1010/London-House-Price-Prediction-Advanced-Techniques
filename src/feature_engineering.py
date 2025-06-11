import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor # For specific model-based imputation
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
from typing import Tuple
import catboost as cb

# --- HELPER FUNCTION (Provided by you) ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def impute_with_knn(data, target_col, predictor_cols):
    """
    使用 KNN 進行缺失值插補
    
    Args:
        data: 包含目標列和預測列的 DataFrame
        target_col: 需要插補的目標列名
        predictor_cols: 用於預測的特徵列名列表
    """
    # 創建 KNN 插補器
    knn = KNNImputer(n_neighbors=5)
    
    # 確保所有預測特徵都是數值型
    numeric_predictors = []
    for col in predictor_cols:
        if data[col].dtype in ['int64', 'float64']:
            numeric_predictors.append(col)
        else:
            # 如果是分類變量，進行標籤編碼
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            numeric_predictors.append(col)
    
    # 先處理預測特徵中的缺失值
    predictor_data = data[numeric_predictors].copy()
    predictor_data = pd.DataFrame(
        knn.fit_transform(predictor_data),
        columns=numeric_predictors,
        index=data.index
    )
    
    # 使用處理後的預測特徵進行目標列的插補
    train_data = predictor_data[~data[target_col].isna()]
    test_data = predictor_data[data[target_col].isna()]
    
    if len(train_data) > 0 and len(test_data) > 0:
        knn = KNNImputer(n_neighbors=5)
        knn.fit(train_data)
        imputed_values = knn.transform(test_data)
        
        # 更新原始數據中的缺失值
        data.loc[data[target_col].isna(), target_col] = imputed_values[:, 0]
    
    return data

# --- MODIFIED IMPUTATION FUNCTION (Using LightGBM) ---
def impute_feature_with_lgbm(
    df_to_impute: pd.DataFrame, 
    col_to_impute: str, 
    predictor_cols_initial: list, 
    is_train: bool, 
    config: dict,
    ohe_categorical_predictors: list = None 
):
    """
    使用 LightGBM (LGBMRegressor) 對單一特徵進行插補。
    處理指定的類別預測變量的 OHE。
    從 config 中存儲/加載模型和 OHE 列。
    """
    df_processed = df_to_impute.copy()
    model_key = f"lgbm_imputer_{col_to_impute}"
    ohe_cols_key_prefix = f"lgbm_imputer_{col_to_impute}_ohe_cols_"
    fallback_value_key = f"{col_to_impute}_fallback_impute_value_lgbm"

    if col_to_impute not in df_processed.columns:
        print(f"警告: 欄位 {col_to_impute} 在數據中未找到，無法進行 LGBM 插補。")
        return df_processed

    missing_mask = df_processed[col_to_impute].isna()
    if not missing_mask.any():
        return df_processed 

    # 初始化預測變量列表
    current_predictor_cols = []
    
    # 添加初始預測變量（排除目標變量）
    for col in predictor_cols_initial:
        if col in df_processed.columns and col != col_to_impute:
            current_predictor_cols.append(col)
    
    # 添加額外的相關特徵（確保不重複）
    additional_features = []
    if 'total_rooms' in df_processed.columns and col_to_impute not in ['bedrooms', 'bathrooms', 'livingRooms'] and 'total_rooms' not in current_predictor_cols:
        additional_features.append('total_rooms')
    
    current_predictor_cols.extend(additional_features)
    
    # 確保預測變量列表中的元素唯一
    current_predictor_cols = list(dict.fromkeys(current_predictor_cols))
    
    df_for_imputation_model = df_processed[current_predictor_cols + ([col_to_impute] if col_to_impute in df_processed else [])].copy()
    
    final_imputation_predictors = []
    # 處理作為預測變量的類別特徵的 OHE
    if ohe_categorical_predictors:
        for cat_col in ohe_categorical_predictors:
            if cat_col in df_for_imputation_model.columns:
                # 確保cat_col是字符串類型，以防pd.get_dummies出錯
                df_for_imputation_model[cat_col] = df_for_imputation_model[cat_col].astype(str)
                dummies = pd.get_dummies(df_for_imputation_model[cat_col], prefix=f"{cat_col}_imp_lgbm", dummy_na=False)
                
                current_ohe_cols_key = f"{ohe_cols_key_prefix}_{cat_col}"
                if is_train:
                    config[current_ohe_cols_key] = dummies.columns.tolist()
                
                learned_ohe_cols = config.get(current_ohe_cols_key, dummies.columns.tolist())
                for learned_col in learned_ohe_cols:
                    if learned_col not in dummies.columns:
                        dummies[learned_col] = 0 # 對齊 OHE 列
                df_for_imputation_model = pd.concat([df_for_imputation_model.drop(columns=[cat_col], errors='ignore'), dummies[learned_ohe_cols]], axis=1)
                final_imputation_predictors.extend(learned_ohe_cols)
            
    # 添加剩餘的非 OHE 預測變量
    final_imputation_predictors.extend([p for p in current_predictor_cols if p not in (ohe_categorical_predictors or [])])
    final_imputation_predictors = sorted(list(set(final_imputation_predictors))) # 排序並確保唯一

    if not final_imputation_predictors:
        print(f"警告: 特徵 '{col_to_impute}' 的LGBM插補模型沒有可用的預測變量。")
        # (後備邏輯與原函數保持一致)
        if is_train:
            median_val = df_processed[col_to_impute].median()
            if pd.isna(median_val): median_val = 0
            config[fallback_value_key] = median_val
            df_processed[col_to_impute] = df_processed[col_to_impute].fillna(median_val)
        else:
            median_val = config.get(fallback_value_key, 0)
            df_processed[col_to_impute] = df_processed[col_to_impute].fillna(median_val)
        return df_processed

    if is_train:
        # 更新 print 語句以顯示使用的特徵總數
        # print(f"訓練 LGBM 插補模型於: {col_to_impute} (使用 {len(final_imputation_predictors)} 個特徵)...")
        train_data_for_imputer = df_for_imputation_model[~missing_mask]
        
        # 移除提前的 NaN 檢查，確保流程總是嘗試訓練模型
        if len(train_data_for_imputer) < 10:
            print(f"警告: 訓練 LGBM 插補模型 {col_to_impute} 的數據不足。使用中位數。")
            median_val = df_processed[col_to_impute].median() # 使用原始df的中位數
            if pd.isna(median_val): median_val = 0
            config[fallback_value_key] = median_val
            df_processed.loc[missing_mask, col_to_impute] = median_val # 只填充缺失部分
            return df_processed

        X_impute_train = train_data_for_imputer[final_imputation_predictors]
        y_impute_train = train_data_for_imputer[col_to_impute]
        
        # 在訓練插補模型前，臨時填充其預測變量中的 NaN，確保模型能順利訓練
        if X_impute_train.isnull().values.any():
            # 使用 0 進行快速填充，也可以考慮中位數
            X_impute_train = X_impute_train.fillna(0)

        # LightGBM 參數 (可根據需要調整)
        lgbm_params = {
            'objective': 'regression_l1', # MAE
            'metric': 'mae',
            'n_estimators': 70, # 較少估計器以加快插補速度
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'num_leaves': 21, # 較少葉子節點
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42,
            'boosting_type': 'gbdt',
        }
        model = lgb.LGBMRegressor(**lgbm_params)
        try:
            model.fit(X_impute_train, y_impute_train)
            config[model_key] = model
            # 使用包含 OHE 特徵的 df_for_imputation_model 進行預測
            df_processed.loc[missing_mask, col_to_impute] = model.predict(df_for_imputation_model.loc[missing_mask, final_imputation_predictors])
        except Exception as e:
            print(f"訓練/預測 LGBM 插補模型 {col_to_impute} 出錯: {e}。使用中位數。")
            median_val = df_processed[col_to_impute].median()
            if pd.isna(median_val): median_val = 0
            config[fallback_value_key] = median_val
            df_processed.loc[missing_mask, col_to_impute] = median_val
            
    else: # is_test/validation
        if model_key in config:
            model = config[model_key]
            try:
                df_processed.loc[missing_mask, col_to_impute] = model.predict(df_for_imputation_model.loc[missing_mask, final_imputation_predictors])
            except Exception as e:
                print(f"使用已訓練的 LGBM 插補模型 {col_to_impute} 出錯: {e}。使用中位數。")
                median_val = config.get(fallback_value_key, 0)
                df_processed.loc[missing_mask, col_to_impute] = median_val
        else:
            print(f"警告: LGBM 插補模型 {model_key} 未在 config 中找到。使用中位數。")
            median_val = config.get(fallback_value_key, 0)
            df_processed.loc[missing_mask, col_to_impute] = median_val
            
    if col_to_impute in ["bedrooms", "bathrooms", "livingRooms"]: 
        df_processed[col_to_impute] = df_processed[col_to_impute].round().fillna(0).astype(int) 
        
    return df_processed

def better_features(features, target_series, is_train, config, columns_to_combine=None, quick_test=False):
    """
    自動化生成和篩選算術組合特徵。
    - 訓練模式 (is_train=True): 探索新特徵，評估其與目標的相關性及與現有特徵的冗餘性，
      並將有效的特徵組合儲存到 config 中。
    - 推論模式 (is_train=False): 從 config 中讀取已儲存的特徵組合，並應用到數據集上。
    """
    print("-> 執行特徵組合...")

    if quick_test:
        print("    啟用快速測試模式，僅使用少量核心特徵進行組合。")
        core_features_for_combination = ['floorAreaSqM', 'total_rooms', 'lat_lon_ratio']
        features_to_use = [f for f in core_features_for_combination if f in features.columns]
    else:
        features_to_use = features.select_dtypes(include=np.number).columns.tolist()

    if len(features_to_use) < 2:
        print("    可用於組合的特徵少於2個，跳過此步驟。")
        return features, config

    if is_train:
        if target_series is None:
            print("    在訓練模式下未提供目標變數，跳過特徵組合。")
            return features, config

        # --- 使用 NumPy 進行向量化計算以大幅提高性能 ---
        print("    初始化 NumPy 矩陣以加速相關性計算...")
        
        target_vector = target_series.values
        target_vector_norm = (target_vector - np.mean(target_vector)) / np.std(target_vector)

        existing_features_matrix = features[features_to_use].values
        
        mean_existing = np.mean(existing_features_matrix, axis=0)
        std_existing = np.std(existing_features_matrix, axis=0)
        std_existing[std_existing == 0] = 1
        normalized_existing_matrix = (existing_features_matrix - mean_existing) / std_existing
        
        n_rows = len(target_vector)
        new_combinations_list = [] 

        print("    正在搜索最佳算術組合 (使用向量化)...")
        
        for i in tqdm(range(len(features_to_use)), desc="    組合特徵搜索", leave=False):
            for j in range(i, len(features_to_use)):
                col1 = features_to_use[i]
                col2 = features_to_use[j]

                for op in ['*', '/', '-', '+']:
                    if col1 == col2 and op in ['-']: continue
                    if i > j and op in ['+', '*']: continue

                    new_feature_name = f'{col1}_{op}_{col2}'
                    
                    if op == '+': new_feature_series = features[col1] + features[col2]
                    elif op == '-': new_feature_series = features[col1] - features[col2]
                    elif op == '*': new_feature_series = features[col1] * features[col2]
                    elif op == '/': new_feature_series = features[col1] / (features[col2] + 1e-9)
                    
                    if new_feature_series.isnull().any() or np.isinf(new_feature_series).any():
                        continue

                    # --- 高效的向量化相關性檢查 ---
                    new_feature_vector = new_feature_series.values
                    
                    mean_new = np.mean(new_feature_vector)
                    std_new = np.std(new_feature_vector)
                    if std_new < 1e-9: continue
                    
                    new_feature_norm = (new_feature_vector - mean_new) / std_new
                    correlation_with_target = np.dot(target_vector_norm, new_feature_norm) / n_rows
                    
                    if abs(correlation_with_target) < 0.05:
                        continue

                    correlations_with_existing = np.dot(new_feature_norm, normalized_existing_matrix) / n_rows
                    if np.max(np.abs(correlations_with_existing)) > 0.95:
                        continue
                    
                    features[new_feature_name] = new_feature_series
                    normalized_existing_matrix = np.c_[normalized_existing_matrix, new_feature_norm]
                    new_combinations_list.append((col1, col2, op))
        
        # 修正後的打印語句
        print(f"\n    發現 {len(new_combinations_list)} 個新組合特徵。")
        config['better_features_list'] = new_combinations_list
        return features, config

    else: # is_train == False
        if 'better_features_list' not in config or not config['better_features_list']:
            print("    在非訓練模式下，未找到已儲存的特徵組合，跳過此步驟。")
            return features, config
        
        combinations = config['better_features_list']
        print(f"    應用 {len(combinations)} 個已儲存的特徵組合。")
        for col1, col2, op in combinations:
            new_col_name = f'{col1}_{op}_{col2}'
            if op == '+': features[new_col_name] = features[col1] + features[col2]
            elif op == '-': features[new_col_name] = features[col1] - features[col2]
            elif op == '*': features[new_col_name] = features[col1] * features[col2]
            elif op == '/': features[new_col_name] = features[col1] / (features[col2] + 1e-9)
        
        return features, config

def select_features_with_lgbm(features, target, k, use_gpu=False):
    """使用 LightGBM 選擇 top-k 特徵，支援 GPU"""
    params = {'random_state': 42, 'n_jobs': -1}
    if use_gpu:
        try:
            params['device'] = 'gpu'
            lgbm = lgb.LGBMRegressor(**params)
            lgbm.fit(features, target)
            print("LGBM feature selection running on GPU.")
        except Exception as e:
            print(f"LGBM on GPU failed ({e}), falling back to CPU.")
            params.pop('device')
            lgbm = lgb.LGBMRegressor(**params)
            lgbm.fit(features, target)
    else:
        lgbm = lgb.LGBMRegressor(**params)
        lgbm.fit(features, target)
    
    feature_importance_df = pd.DataFrame({
        'feature': features.columns,
        'importance': lgbm.feature_importances_
    }).sort_values('importance', ascending=False)
    return feature_importance_df.head(k)['feature'].tolist()

def select_features_with_xgb(features, target, k, use_gpu=False):
    """使用 XGBoost 選擇 top-k 特徵，支援 GPU"""
    params = {'random_state': 42, 'n_jobs': -1}
    # 確保特徵名稱不包含 XGBoost 不支援的字符
    safe_features = features.rename(columns = lambda x: x.replace('[', '').replace(']', '').replace('<', ''))
    
    if use_gpu:
        try:
            params['tree_method'] = 'gpu_hist'
            xgb_model = xgb.XGBRegressor(**params)
            xgb_model.fit(safe_features, target)
            print("XGB feature selection running on GPU.")
        except Exception as e:
            print(f"XGB on GPU failed ({e}), falling back to CPU.")
            params.pop('tree_method')
            xgb_model = xgb.XGBRegressor(**params)
            xgb_model.fit(safe_features, target)
    else:
        xgb_model = xgb.XGBRegressor(**params)
        xgb_model.fit(safe_features, target)

    feature_importance_df = pd.DataFrame({
        'feature': safe_features.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 將安全名稱映射回原始名稱
    top_k_safe_cols = feature_importance_df.head(k)['feature'].tolist()
    return [features.columns[safe_features.columns.get_loc(col)] for col in top_k_safe_cols]

def select_features_with_cat(features, target, k, use_gpu=False):
    """使用 CatBoost 選擇 top-k 特徵，支援 GPU"""
    params = {'random_state': 42, 'verbose': 0}
    if use_gpu:
        try:
            params['task_type'] = 'GPU'
            cat_model = cb.CatBoostRegressor(**params)
            cat_model.fit(features, target)
            print("CatBoost feature selection running on GPU.")
        except Exception as e:
            print(f"CatBoost on GPU failed ({e}), falling back to CPU.")
            params.pop('task_type')
            cat_model = cb.CatBoostRegressor(**params)
            cat_model.fit(features, target)
    else:
        cat_model = cb.CatBoostRegressor(**params)
        cat_model.fit(features, target)

    feature_importance_df = pd.DataFrame({
        'feature': features.columns,
        'importance': cat_model.feature_importances_
    }).sort_values('importance', ascending=False)
    return feature_importance_df.head(k)['feature'].tolist()

def select_features(
    X: pd.DataFrame, 
    y: pd.Series, 
    config: dict,
    k: int = 40,
    use_gpu: bool = False,
) -> (list, dict):
    """
    使用三種不同的模型（LGBM、XGBoost、CatBoost）來選擇 top-k 特徵。
    返回三種模型選擇的特徵的聯集。
    """
    print(f"\n=== 預處理步驟 2: 預先執行特徵選擇 ===")

    # 如果總特徵數小於等於 k，則直接返回所有特徵，無需選擇
    if X.shape[1] <= k:
        print(f"特徵總數 ({X.shape[1]}) 小於等於 k ({k})，跳過特徵選擇。")
        selected_features = X.columns.tolist()
        config['selected_features'] = selected_features
        return selected_features, config

    lgb_features = select_features_with_lgbm(X.copy(), y, k, use_gpu)
    print(f"LGB 選擇了 {len(lgb_features)} 個 top-{k} 特徵")

    xgb_features = select_features_with_xgb(X.copy(), y, k, use_gpu)
    print(f"XGB 選擇了 {len(xgb_features)} 個 top-{k} 特徵")

    cat_features = select_features_with_cat(X.copy(), y, k, use_gpu)
    print(f"CatBoost 選擇了 {len(cat_features)} 個 top-{k} 特徵")

    # 合併所有選中的特徵
    selected_features = list(set(lgb_features + xgb_features + cat_features))
    
    # 如果沒有選中任何特徵，使用所有特徵
    if not selected_features:
        selected_features = X.columns.tolist()
    
    print(f"最終選擇了 {len(selected_features)} 個特徵")
    print(f"選中的特徵列表: {selected_features}")
    
    # 保存選中的特徵到配置中
    config['selected_features'] = selected_features
    
    return selected_features, config

def scale_features(df, is_train=True, config=None):
    """
    特徵縮放函數
    
    Args:
        df: 輸入數據框
        is_train: 是否為訓練集
        config: 配置字典
    
    Returns:
        縮放後的特徵數據框和更新後的配置
    """
    if config is None:
        config = {}
    
    # 1. 處理無限值和極大值
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    # 2. 選擇需要縮放的特徵
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    
    if is_train:
        # 訓練集：創建新的 scaler
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
        config['scaler'] = scaler
        config['scaled_features'] = numeric_features.tolist()
    else:
        # 測試集：使用訓練集的 scaler
        if 'scaler' in config and 'scaled_features' in config:
            # 確保所有需要的特徵都存在
            missing_features = set(config['scaled_features']) - set(df.columns)
            if missing_features:
                print(f"警告：缺少以下特徵：{missing_features}")
                # 為缺失的特徵添加零值列
                for feature in missing_features:
                    df[feature] = 0
            
            # 使用訓練集的 scaler 進行轉換
            df[config['scaled_features']] = config['scaler'].transform(df[config['scaled_features']])
        else:
            print("警告：未找到訓練集的 scaler，跳過特徵縮放")
    
    return df, config

def validate_features(df, is_train=True, config=None):
    """
    特徵驗證函數
    Args:
        df: 輸入數據框
        is_train: 是否為訓練集
        config: 配置字典
    Returns:
        驗證後的特徵數據框
    """
    if config is None:
        config = {}
    # 1. 處理無限值
    df = df.replace([np.inf, -np.inf], np.nan)
    # 2. 處理異常值（使用 IQR 方法）
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # 將異常值替換為邊界值
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    # 3. 處理剩餘的 NaN
    df = df.fillna(0)
    return df

def handle_missing_values(df: pd.DataFrame, is_train: bool, config: dict) -> pd.DataFrame:
    """
    使用迭代策略處理 DataFrame 中的缺失值。
    對於訓練集，它會進行多輪 LGBM 插補。
    """
    features = df.copy()
    
    # 找出所有需要插補的數值型特徵
    numeric_cols = features.select_dtypes(include=np.number).columns.tolist()
    cols_to_impute = [col for col in numeric_cols if features[col].isnull().any()]

    if not cols_to_impute:
        return features

    if is_train:
        print(f"\n開始迭代式插補，目標欄位: {cols_to_impute}")
        # 進行多輪迭代插補
        n_iterations = 3  # 設置迭代次數
        for i in range(n_iterations):
            print(f"--- 插補迭代 [第 {i + 1}/{n_iterations} 輪] ---")
            for col in cols_to_impute:
                if features[col].isnull().any():
                    # 預測變量為所有其他的數值型欄位
                    predictor_cols = [c for c in numeric_cols if c != col]
                    features = impute_feature_with_lgbm(features, col, predictor_cols, is_train, config)
    else:
        # 對於測試集，使用訓練時儲存的中位數或簡單策略
        print("\n為測試集處理缺失值...")
        for col in cols_to_impute:
            # 使用在訓練階段可能保存的中位數，或計算當前測試集的中位數作為後備
            fallback_median = features[col].median()
            median_val = config.get(f'{col}_fallback_impute_value_lgbm', fallback_median)
            features[col] = features[col].fillna(median_val)
            if features[col].isnull().any(): # 如果仍然有 NaN
                features[col] = features[col].fillna(0)

    return features

def process_categorical_features(df: pd.DataFrame, is_train: bool, config: dict) -> pd.DataFrame:
    """
    Process categorical features in the DataFrame.
    
    Args:
        df: Input DataFrame
        is_train: Whether this is training data
        config: Configuration dictionary
        
    Returns:
        DataFrame with processed categorical features
    """
    features = df.copy()
    
    # 處理分類特徵
    categorical_features = features.select_dtypes(include=['object']).columns
    for col in categorical_features:
        if features[col].isnull().any():
            features[col] = features[col].fillna('missing')
        
        # 使用頻率編碼
        if is_train:
            # 計算每個類別的頻率
            value_counts = features[col].value_counts(normalize=True)
            # 保存頻率映射到配置中
            config[f'freq_encoder_{col}'] = value_counts.to_dict()
            # 使用頻率編碼
            features[col] = features[col].map(value_counts)
        else:
            # 使用訓練集的頻率映射
            freq_map = config.get(f'freq_encoder_{col}', {})
            # 對於未見過的類別，使用最小頻率
            min_freq = min(freq_map.values()) if freq_map else 0
            features[col] = features[col].map(lambda x: freq_map.get(x, min_freq))
    
    return features

def engineer_features(features: pd.DataFrame, is_train: bool, config: dict, quick_test: bool = False):
    """
    對給定的 DataFrame 執行完整的特徵工程流程。
    
    Args:
        features: 包含特徵的 DataFrame
        is_train: 布林值，指示是否為訓練模式
        config: 用於存儲/加載狀態的配置字典
        quick_test: 布林值，指示是否啟用快速測試模式
    
    Returns:
        處理後的 DataFrame 和 target_series (如果是在訓練模式)
    """
    if config is None:
        config = {}

    # 在所有操作之前，先將 price 分離出來
    target_series = None
    if is_train and 'price' in features.columns:
        # 對 price 進行對數轉換以處理偏態
        target_series = np.log1p(features['price'])
        # 將原始 price 從特徵中移除，避免其參與任何特徵工程
        features = features.drop(columns=['price'])
        print(f"初始特徵數量 (移除 price 後): {features.shape[1]}")
    elif 'price' in features.columns:
        # 如果是驗證/測試集，但仍有 price 欄位，也將其移除
        features = features.drop(columns=['price'])
        print(f"初始特徵數量 (移除 price 後): {features.shape[1]}")
    else:
        print(f"初始特徵數量: {features.shape[1]}")

    # 執行所有特徵工程步驟...
    features = create_derived_features(features, is_train=is_train, config=config)
    print(f"-> 衍生特徵後數量: {features.shape[1]}")

    features = process_categorical_features(features, is_train=is_train, config=config)
    print(f"-> 類別特徵處理後數量: {features.shape[1]}")

    features = handle_missing_values(features, is_train=is_train, config=config)
    print(f"-> 缺失值處理後數量: {features.shape[1]}")
    
    # 執行自動化特徵組合
    features, config = better_features(features, target_series=target_series, is_train=is_train, config=config, quick_test=quick_test)
    print(f"-> 特徵組合後數量: {features.shape[1]}")

    # 正確處理 scale_features 的返回
    features, config = scale_features(features, is_train=is_train, config=config)
    print(f"-> 特徵縮放後數量: {features.shape[1]}")
    
    # 執行特徵驗證
    features = validate_features(features, is_train=is_train, config=config)
    print(f"-> 特徵驗證後數量: {features.shape[1]}")
    
    # 在所有特徵工程完成後，如果是在訓練模式，則將處理過的 target 加回去
    if is_train and target_series is not None:
        # features[config.get('target_col', 'price')] = target_series
        return features, target_series, config
    
    # 對於 is_train=False 的情況，也返回三個值以保持一致性
    return features, None, config

def create_derived_features(features: pd.DataFrame, is_train: bool, config: dict) -> pd.DataFrame:
    """創建衍生特徵"""
    
    # --- 房間相關特徵 ---
    room_cols = ['bathrooms', 'bedrooms', 'livingRooms']
    if all(col in features.columns for col in room_cols):
        features['total_rooms'] = features[room_cols].sum(axis=1)
        if 'floorAreaSqM' in features.columns and features['floorAreaSqM'].notna().any():
             # 避免除以零
            features['rooms_per_area'] = features['total_rooms'] / (features['floorAreaSqM'] + 1e-6)
        
        # 避免除以零
        features['bathrooms_ratio'] = features['bathrooms'] / (features['total_rooms'] + 1e-6)
        features['bedrooms_ratio'] = features['bedrooms'] / (features['total_rooms'] + 1e-6)
        features['livingRooms_ratio'] = features['livingRooms'] / (features['total_rooms'] + 1e-6)

    # --- 地理位置特徵 ---
    geo_cols = ['latitude', 'longitude']
    if all(col in features.columns for col in geo_cols):
        features['lat_lon_ratio'] = features['latitude'] / (features['longitude'] + 1e-9)
        features['lat_lon_product'] = features['latitude'] * features['longitude']

    # --- 時間相關特徵 ---
    if 'sale_month' in features.columns:
        features['sale_month'] = pd.to_numeric(features['sale_month'], errors='coerce')
        features['sin_month'] = np.sin(2 * np.pi * features['sale_month'] / 12)
        features['cos_month'] = np.cos(2 * np.pi * features['sale_month'] / 12)
        features['months_since_start'] = (features['sale_year'] - features['sale_year'].min()) * 12 + features['sale_month']
        
        # 季節特徵
        features['season'] = features['sale_month'].apply(lambda x: (x % 12 + 3) // 3)
        features['is_spring'] = (features['season'] == 1).astype(int)
        features['is_summer'] = (features['season'] == 2).astype(int)

    # --- 為關鍵欄位的缺失值創建指示符特徵 ---
    # 定義我們關心是否缺失的欄位列表
    cols_to_check_missing = ['bathrooms', 'bedrooms', 'livingRooms', 'floorAreaSqM']
    
    # 只對數據中實際存在的欄位進行操作
    for col in cols_to_check_missing:
        if col in features.columns:
            features[f'{col}_is_missing'] = features[col].isna().astype(int)
            
    # --- 面積的對數轉換 ---
    # 檢查 'floorAreaSqM' 是否存在，以安全地進行對數轉換
    if 'floorAreaSqM' in features.columns:
        features['floorAreaSqM_log'] = np.log1p(features['floorAreaSqM'])

    return features

def find_feature_combinations(features: pd.DataFrame, target: pd.Series, quick_test: bool = False, config: dict = None) -> (pd.DataFrame, list):
    # ... existing code ...
    pass