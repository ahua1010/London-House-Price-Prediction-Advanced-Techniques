# -*- coding: utf-8 -*-
# # IMPORTS

# ## 匯入必要的函式庫
import numpy as np
import pandas as pd
import joblib # 用於儲存/載入模型
import pickle # <--- 新增
import os     # 用於操作文件路徑和目錄
from prettytable import PrettyTable # 用於以表格形式顯示資訊
from tqdm import tqdm # 進度條
from functools import partial
from itertools import combinations
from random import randint, uniform
import gc # 垃圾回收
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, PowerTransformer, FunctionTransformer
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss # 移除了部分未使用指標
import lightgbm as lgb
import lightgbm.callback as lgb_callback # 修正：明確匯入 callback
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.decomposition import TruncatedSVD # 用於transformer函數

import warnings
warnings.filterwarnings("ignore") # 忽略警告訊息
pd.pandas.set_option('display.max_columns', None) # 顯示所有欄位

# --- 全局設定區 ---
# 模式設定: True = 執行訓練和儲存; False = 載入已儲存模型進行預測
TRAIN_MODE = True # <--- ****** 修改這裡來切換模式 ******

# 檔案路徑設定 (根據模式可能需要不同的檔案)
TRAIN_FILE = './train.csv'
TEST_FILE_FOR_TRAINING = './test.csv' # 訓練模式下用於評估或生成提交檔的測試集
ORIGINAL_FILE = "./train_extend.csv" # 訓練模式下使用的額外資料
SAMPLE_SUBMISSION_FILE = './sample_submission.csv' # 訓練模式下用於生成提交檔
TEST_FILE_FOR_PREDICTION = './new_data_to_predict.csv' # <--- 預測模式下讀取的新數據檔案
OUTPUT_FILE_PREDICTION = './new_data_predictions_stacking.csv' # <--- 預測模式下的輸出檔名
OUTPUT_FILE_TRAINING_SUB = 'submission_stacking_train_mode.csv' # <--- 訓練模式下的輸出檔名

# 模型與物件儲存路徑
MODEL_SAVE_DIR = './saved_stacking_models/' # <--- 模型儲存目錄
SCALER_FILE = os.path.join(MODEL_SAVE_DIR, 'scaler.pkl') # <--- Scaler 儲存路徑
META_MODEL_FILE = os.path.join(MODEL_SAVE_DIR, 'meta_model.pkl') # <--- 元模型儲存路徑
FINAL_FEATURES_FILE = os.path.join(MODEL_SAVE_DIR, 'final_feature_list.pkl') # <--- 最終特徵列表儲存路徑
INDICES_FILE_PATH = 'selected_extend_indices.pkl' # <--- 新增：設定篩選索引檔案的路徑
TOP_FEATURES_FILE = os.path.join(MODEL_SAVE_DIR, 'top_original_features.pkl') # <--- 新增：儲存 Top 15 特徵列表

# 交叉驗證與模型設定 (兩模式通用)
N_SPLITS = 5
RANDOM_STATE_LIST = [42] # 為了演示和速度，只用一個 seed，您可以增加
N_ESTIMATORS = 2000 # 基礎模型迭代次數 (配合早停)
EARLY_STOPPING_ROUNDS = 100 # 早停輪數
ENABLE_BETTER_FEATURES = True # 是否啟用動態算術特徵搜索 (僅訓練模式)
N_FEATURES_TO_SELECT = 50 # 特徵選擇數量 (兩模式通用)
TOP_N_ORIGINAL_FEATURES = 15 # <--- 新增：選擇 Top 15 原始特徵

# --- 修改：設定 ExtraTrees 為元模型 ---
META_MODEL_CLASS = ExtraTreesClassifier
META_MODEL_PARAMS = {
    'n_estimators': 200,       # ET 的樹數量
    'max_depth': 7,            # 限制深度防過擬合
    'min_samples_split': 8,    # 內部節點最小樣本數
    'min_samples_leaf': 4,     # 葉節點最小樣本數
    'n_jobs': -1,
    'random_state': 42,
    'class_weight': 'balanced', # 嘗試平衡類別
    'verbose': 0
}
# --- 修改結束 ---


global device, verbose
device = 'cpu' # 或 'gpu'
verbose = False
# --- 全局設定區結束 ---


# --- 特徵工程與輔助函數定義 ---

def create_extra_features(df):
    """
    對特定欄位進行調整與限制範圍。
    (保持原邏輯不變)
    """
    # 處理聽力 hearing(left), hearing(right)
    best_hearing = np.where(df['hearing(left)'] < df['hearing(right)'], df['hearing(left)'], df['hearing(right)'])
    worst_hearing = np.where(df['hearing(left)'] < df['hearing(right)'], df['hearing(right)'], df['hearing(left)'])
    df['hearing(left)'] = best_hearing - 1
    df['hearing(right)'] = worst_hearing - 1
    # 處理視力 eyesight(left), eyesight(right)
    df['eyesight(left)'] = np.where(df['eyesight(left)'] > 9, 0, df['eyesight(left)'])
    df['eyesight(right)'] = np.where(df['eyesight(right)'] > 9, 0, df['eyesight(right)'])
    best_eyesight = np.where(df['eyesight(left)'] < df['eyesight(right)'], df['eyesight(left)'], df['eyesight(right)'])
    worst_eyesight = np.where(df['eyesight(left)'] < df['eyesight(right)'], df['eyesight(right)'], df['eyesight(left)'])
    df['eyesight(left)'] = best_eyesight
    df['eyesight(right)'] = worst_eyesight
    # 對部分數值特徵進行範圍限制 (Clip)
    df['Gtp'] = np.clip(df['Gtp'], 0, 300)
    df['HDL'] = np.clip(df['HDL'], 0, 110)
    df['LDL'] = np.clip(df['LDL'], 0, 200)
    df['ALT'] = np.clip(df['ALT'], 0, 150)
    df['AST'] = np.clip(df['AST'], 0, 100)
    df['serum creatinine'] = np.clip(df['serum creatinine'], 0, 3)
    return df

def min_max_scaler(df1, df2, column):
    """
    對指定欄位進行 Min-Max 標準化 (同時作用於兩個 DataFrame)。
    (保持原邏輯不變, 但參數名改為 df1, df2)
    """
    max_val = max(df1[column].max(), df2[column].max())
    min_val = min(df1[column].min(), df2[column].min())
    epsilon = 1e-9 # 避免除以零
    scale = max_val - min_val + epsilon
    df1[column] = (df1[column] - min_val) / scale
    df2[column] = (df2[column] - min_val) / scale
    return df1, df2

def OHE(train_df, test_df, cols, target):
    """
    對指定類別欄位進行 One-Hot Encoding。
    (保持原邏輯不變)
    """
    combined = pd.concat([train_df, test_df], axis=0)
    initial_cols = combined.columns.tolist() # 記錄原始欄位
    for col in cols:
        if col not in combined.columns: continue # 如果欄位已被移除則跳過
        one_hot = pd.get_dummies(combined[col], prefix=col, prefix_sep='_OHE_')
        counts = combined[col].value_counts()
        if len(counts) > 1:
            min_count_category = counts.idxmin()
            try:
                 one_hot = one_hot.drop(f"{col}_OHE_{min_count_category}", axis=1)
            except KeyError:
                 print(f"警告：無法在 OHE 後移除欄位 {col}_OHE_{min_count_category}")
                 pass
        combined = pd.concat([combined, one_hot], axis="columns")
        # combined = combined.loc[:, ~combined.columns.duplicated()] # 可能移除需要的欄位，先註解
        combined = combined.drop(columns=[col])

    # 分割回訓練集和測試集
    train_ohe = combined.iloc[:len(train_df)].copy()
    test_ohe = combined.iloc[len(train_df):].copy()
    test_ohe.reset_index(inplace=True, drop=True)
    if target in test_ohe.columns:
        test_ohe.drop(columns=[target], inplace=True)
    # 確保測試集有訓練集的所有欄位（填充 0）
    missing_cols_in_test = list(set(train_ohe.columns) - set(test_ohe.columns) - set([target]))
    for c in missing_cols_in_test:
         test_ohe[c] = 0
    # 確保欄位順序一致
    test_ohe = test_ohe[train_ohe.drop(columns=[target], errors='ignore').columns]

    return train_ohe, test_ohe

# --- 修正：定義 lgb_params_impute ---
lgb_params_impute = {
    'objective': 'regression_l2', 'metric': 'rmse', 'n_estimators': 100,
    'learning_rate': 0.05, 'feature_fraction': 0.8, 'subsample': 0.8,
    'num_leaves': 31, 'max_depth': -1, 'seed': 42, 'n_jobs': -1,
    'verbose': -1, 'boosting_type': 'gbdt',
    # 'device': device # 可以取消註解
}
# --- 修正結束 ---

def rmse(y1, y2):
    """計算 Root Mean Squared Error (RMSE)"""
    return np.sqrt(mean_squared_error(np.array(y1), np.array(y2)))

def store_missing_rows(df, features):
    """儲存包含缺失值的行索引，以便後續填補"""
    missing_rows_indices = {}
    for feature in features:
        missing_rows_indices[feature] = df[df[feature].isnull()].index
    return missing_rows_indices

def fill_missing_numerical(train, test, target, max_iterations=10):
    """
    使用 LightGBM 迭代填補數值特徵的缺失值 (修正版：確保僅使用數值特徵)。
    (使用修正後的邏輯)
    """
    print("開始執行迭代式數值缺失值填補...")
    train_temp = train.copy()
    if target in train_temp.columns:
        train_temp = train_temp.drop(columns=[target])

    train_index = train.index
    test_index = test.index

    df = pd.concat([train_temp, test], axis="rows", ignore_index=True)

    all_numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    if target in all_numeric_features: all_numeric_features.remove(target)

    features_to_impute = [f for f in df.columns if df[f].isna().sum() > 0 and f in all_numeric_features]

    if len(features_to_impute) > 0:
        print(f"開始迭代填補以下數值特徵的缺失值: {features_to_impute}")
        missing_rows_indices = store_missing_rows(df, features_to_impute)

        print("進行初始平均值填補...")
        for f in features_to_impute:
            mean_val = df[f].mean()
            if pd.isna(mean_val): mean_val = 0 # 如果整列都是 NaN，用 0 填充
            df[f] = df[f].fillna(mean_val)
            print(f"  特徵 '{f}' 用平均值 {mean_val:.4f} 填補了 {len(missing_rows_indices[f])} 個缺失值。")

        dictionary = {feature: [] for feature in features_to_impute}

        for iteration in tqdm(range(max_iterations), desc="迭代填補缺失值"):
            valid_numeric_predictors = [col for col in all_numeric_features if col in df.columns and col != target]

            for feature in features_to_impute:
                rows_miss = missing_rows_indices[feature]
                if rows_miss.empty: continue

                non_missing_rows = df.index.difference(rows_miss)
                current_predictors = [p for p in valid_numeric_predictors if p != feature]

                if not current_predictors:
                     print(f"警告：特徵 '{feature}' 沒有有效的數值預測變數，跳過。")
                     dictionary[feature].append(np.nan)
                     continue

                X_train_impute = df.loc[non_missing_rows, current_predictors]
                y_train_impute = df.loc[non_missing_rows, feature]
                missing_temp_features = df.loc[rows_miss, current_predictors]

                if y_train_impute.nunique(dropna=False) <= 1: # 檢查是否常數或全 NaN
                    print(f"警告：特徵 '{feature}' 的非缺失值只有一個唯一值或無值，無法訓練填補模型。")
                    dictionary[feature].append(np.nan)
                    continue

                y_pred_prev = df.loc[rows_miss, feature].copy()

                model = lgb.LGBMRegressor(**lgb_params_impute)
                try:
                    # 使用訓練集子集做早停驗證
                    train_sub_idx = X_train_impute.index[:len(X_train_impute)//5]
                    eval_set = [(X_train_impute.loc[train_sub_idx], y_train_impute.loc[train_sub_idx])]
                    model.fit(X_train_impute, y_train_impute, eval_set=eval_set,
                              callbacks=[lgb_callback.early_stopping(5, verbose=False)])
                except Exception as e:
                     print(f"\n錯誤：在訓練填補模型時遇到問題 (特徵: {feature}) - {e}")
                     dictionary[feature].append(np.nan)
                     continue

                try:
                    y_pred = model.predict(missing_temp_features)
                    df.loc[rows_miss, feature] = y_pred
                    # 檢查 y_pred 和 y_pred_prev 是否包含 NaN
                    mask = ~np.isnan(y_pred) & ~np.isnan(y_pred_prev)
                    if mask.sum() > 0: # 只有在有非 NaN 值可比較時才計算 RMSE
                        error_minimize = rmse(y_pred[mask], y_pred_prev[mask])
                        dictionary[feature].append(error_minimize)
                    else:
                        dictionary[feature].append(np.nan)
                except Exception as e:
                     print(f"\n錯誤：在預測缺失值時遇到問題 (特徵: {feature}) - {e}")
                     dictionary[feature].append(np.nan)
                     continue

        # 更新原始 DataFrame
        train_rows_in_df = df.iloc[:len(train)]
        test_rows_in_df = df.iloc[len(train):]
        train.loc[train_index, features_to_impute] = train_rows_in_df[features_to_impute].values
        test.loc[test_index, features_to_impute] = test_rows_in_df[features_to_impute].values

        print("迭代填補完成。")
    else:
        print("數據框中無數值特徵需要填補缺失值。")

    return train, test

def transformer(train, test, cont_cols, target):
    """
    對指定的連續數值欄位應用多種轉換（log, sqrt, box-cox, yeo-johnson, power）。
    (修改版：暫不使用PCA，並保留所有成功產生的轉換特徵，不進行評估篩選)
    Args:
        train (pd.DataFrame): 訓練集。
        test (pd.DataFrame): 測試集。
        cont_cols (list): 要進行轉換的連續數值欄位列表。
        target (str): 目標變數名稱。
    Returns:
        tuple: 轉換後的 (訓練集, 測試集)。
    """
    # --- 修改：移除不再需要的 global 變數和表格 ---
    # global unimportant_features # 不再篩選，不需要記錄
    # global overall_best_score   # 不再評估，不需要記錄
    # global overall_best_col   # 不再評估，不需要記錄
    train_copy = train.copy()
    test_copy = test.copy()
    # table = PrettyTable()       # 不再評估，不需要表格
    # table.field_names = ['原始特徵', '原始 ROC_AUC', '最佳轉換', '轉換後 ROC_AUC']
    print("開始應用多種數值轉換 (保留所有成功結果, 暫不使用 PCA)...")

    all_successfully_added_cols = [] # 追蹤所有成功添加的欄位名

    for col in tqdm(cont_cols, desc="處理數值特徵轉換"):

        # 移除先前可能存在的舊轉換欄位 (保持不變)
        original_col_exists_train = col in train_copy.columns
        original_col_exists_test = col in test_copy.columns
        if not (original_col_exists_train and original_col_exists_test):
             print(f"警告：原始欄位 {col} 在 train 或 test 中不存在，跳過此欄位的轉換。")
             continue

        for c in ["log_"+col, "sqrt_"+col, "bx_cx_"+col, "y_J_"+col, "log_sqrt"+col, "pow_"+col, "pow2_"+col, col+"_pca_comb"]:
            if c in train_copy.columns: train_copy = train_copy.drop(columns=[c])
            if c in test_copy.columns: test_copy = test_copy.drop(columns=[c])

        # --- (各種轉換的 try...except 區塊) ---
        # 在每個 try 區塊成功執行後，將新欄位名加入 all_successfully_added_cols
        # Log 轉換 (log1p)
        try:
            new_col_name = "log_"+col
            train_copy[new_col_name] = np.log1p(train_copy[col])
            test_copy[new_col_name] = np.log1p(test_copy[col])
            all_successfully_added_cols.append(new_col_name)
        except Exception as e: print(f"警告：對欄位 {col} 進行 Log 轉換失敗 - {e}")

        # 平方根轉換 (sqrt)
        try:
            new_col_name = "sqrt_"+col
            # 檢查是否有負值
            train_neg = (train_copy[col] < 0).any()
            test_neg = (test_copy[col] < 0).any()
            if not (train_neg or test_neg):
                 # 確保 Series 是 float 類型以避免 sqrt 問題
                 train_copy[new_col_name] = np.sqrt(train_copy[col].astype(float))
                 test_copy[new_col_name] = np.sqrt(test_copy[col].astype(float))
                 all_successfully_added_cols.append(new_col_name)
            else: print(f"警告：欄位 {col} 包含負值，跳過 Sqrt 轉換。")
        except Exception as e: print(f"警告：對欄位 {col} 進行 Sqrt 轉換失敗 - {e}")

        # Box-Cox 轉換
        try:
            new_col_name = "bx_cx_"+col
            # Box-Cox 要求所有值為正，即使加了 epsilon 也可能失敗
            min_train_val = train_copy[col].min()
            min_test_val = test_copy[col].min()
            epsilon = 1e-6 # 嘗試更小的 epsilon
            if min_train_val + epsilon > 0 and min_test_val + epsilon > 0:
                transformer_bc = PowerTransformer(method='box-cox', standardize=False) # 不進行標準化
                try:
                    # 分開 fit 和 transform 可能更穩定
                    transformer_bc.fit(train_copy[[col]] + epsilon)
                    train_copy[new_col_name] = transformer_bc.transform(train_copy[[col]] + epsilon)
                    test_copy[new_col_name] = transformer_bc.transform(test_copy[[col]] + epsilon)
                    all_successfully_added_cols.append(new_col_name)
                except ValueError as ve: print(f"警告：對欄位 {col} 進行 Box-Cox 擬合/轉換失敗 - {ve}")
            else: print(f"警告：欄位 {col} 包含非正數（即使加 epsilon），跳過 Box-Cox 轉換。")
        except Exception as e: print(f"警告：對欄位 {col} 進行 Box-Cox 轉換時發生意外錯誤 - {e}")

        # Yeo-Johnson 轉換
        try:
            new_col_name = "y_J_"+col
            transformer_yj = PowerTransformer(method='yeo-johnson', standardize=False) # 不進行標準化
            try:
                transformer_yj.fit(train_copy[[col]]) # 在訓練集上擬合
                train_copy[new_col_name] = transformer_yj.transform(train_copy[[col]])
                test_copy[new_col_name] = transformer_yj.transform(test_copy[[col]])
                all_successfully_added_cols.append(new_col_name)
            except ValueError as ve: print(f"警告：對欄位 {col} 進行 Yeo-Johnson 擬合/轉換失敗 - {ve}")
        except Exception as e: print(f"警告：對欄位 {col} 進行 Yeo-Johnson 轉換時發生意外錯誤 - {e}")

        # Power 轉換 (0.25 次方)
        try:
            new_col_name = "pow_"+col
            min_val_train = train_copy[col].min()
            # 對 train 和 test 分別平移 (避免數據洩漏)
            power_transform_025_train = lambda x: np.power(np.maximum(0, x + 1 - min_val_train), 0.25) # 確保非負
            power_transform_025_test = lambda x: np.power(np.maximum(0, x + 1 - min_val_train), 0.25) # 用 train 的 min 平移 test
            transformer_p025 = FunctionTransformer(power_transform_025_train, validate=False)
            train_copy[new_col_name] = transformer_p025.fit_transform(train_copy[[col]])
            transformer_p025_test = FunctionTransformer(power_transform_025_test, validate=False)
            test_copy[new_col_name] = transformer_p025_test.transform(test_copy[[col]])
            all_successfully_added_cols.append(new_col_name)
        except Exception as e: print(f"警告：對欄位 {col} 進行 Power(0.25) 轉換失敗 - {e}")

        # Power 轉換 (2 次方)
        try:
            new_col_name = "pow2_"+col
            min_val_train = train_copy[col].min()
            # 分別平移
            power_transform_2_train = lambda x: np.power(np.maximum(0, x + 1 - min_val_train), 2)
            power_transform_2_test = lambda x: np.power(np.maximum(0, x + 1 - min_val_train), 2)
            transformer_p2 = FunctionTransformer(power_transform_2_train, validate=False)
            train_copy[new_col_name] = transformer_p2.fit_transform(train_copy[[col]])
            transformer_p2_test = FunctionTransformer(power_transform_2_test, validate=False)
            test_copy[new_col_name] = transformer_p2_test.transform(test_copy[[col]])
            all_successfully_added_cols.append(new_col_name)
        except Exception as e: print(f"警告：對欄位 {col} 進行 Power(2) 轉換失敗 - {e}")

        # Log + Sqrt 轉換
        # 檢查 sqrt 是否成功生成 (基於欄位名)
        sqrt_col_name = "sqrt_" + col
        if sqrt_col_name in train_copy.columns and sqrt_col_name in test_copy.columns:
             try:
                 new_col_name = "log_sqrt"+col
                 train_copy[new_col_name] = np.log1p(train_copy[sqrt_col_name])
                 test_copy[new_col_name] = np.log1p(test_copy[sqrt_col_name])
                 all_successfully_added_cols.append(new_col_name)
             except Exception as e: print(f"警告：對欄位 {col} 進行 Log(Sqrt) 轉換失敗 - {e}")
        # --- 各種轉換結束 ---

        # --- 移除：不再需要在循環內填補 ---
        # train_copy, test_copy = fill_missing_numerical(train_copy, test_copy, target, 5)

        # --- 修改：移除 PCA 區塊 ---
        # (整個 PCA 的 try...except 區塊被移除)
        # --- 修改結束 ---

        # --- 移除：不再需要在循環內重設索引 ---
        # test_copy = test_copy.reset_index(drop=True)

        # --- 修改：移除評估和篩選區塊 ---
        # (整個 KFold 迴圈、計算 auc_scores、找 best_col、刪除 cols_to_drop 的邏輯被移除)
        # --- 修改結束 ---

    # --- 循環結束後 ---

    print("\n數值轉換完成 (保留所有成功結果)。")
    print(f"總共成功添加了 {len(list(set(all_successfully_added_cols)))} 個新的轉換特徵欄位。")

    # --- 新增：在函數末尾進行一次 NaN 填補 ---
    print("對所有轉換後特徵進行最終 NaN 填補...")
    # 填補所有數值欄位，因為轉換可能引入 NaN
    numeric_cols_final = train_copy.select_dtypes(include=np.number).columns.tolist()
    if target in numeric_cols_final: numeric_cols_final.remove(target)
    # 調用 fill_missing_numerical (假設它能處理所有數值列)
    # 注意：這裡的 train_copy, test_copy 可能包含 target 列，fill_missing_numerical 內部會處理
    train_copy, test_copy = fill_missing_numerical(train_copy, test_copy, target, 5)
    print("最終 NaN 填補完成。")
    # --- 新增結束 ---

    # --- 新增：確保測試集欄位與訓練集一致 ---
    print("確保測試集欄位與訓練集一致...")
    train_cols_final = [c for c in train_copy.columns if c != target]
    # 確保 test_copy 只包含 train_cols_final 中的欄位，並且順序一致
    missing_cols_in_test_final = list(set(train_cols_final) - set(test_copy.columns))
    if missing_cols_in_test_final:
         print(f"  在新測試集中添加缺失欄位: {missing_cols_in_test_final} (填充0)")
         for c in missing_cols_in_test_final:
             test_copy[c] = 0 # 使用 0 填充 (或考慮其他策略)

    extra_cols_in_test_final = list(set(test_copy.columns) - set(train_cols_final))
    if extra_cols_in_test_final:
        print(f"  從測試集中移除多餘欄位: {extra_cols_in_test_final}")
        test_copy = test_copy.drop(columns=extra_cols_in_test_final)

    # 按訓練集順序排列
    test_copy = test_copy[train_cols_final]
    print("欄位一致性檢查完成。")
    # --- 新增結束 ---

    return train_copy, test_copy


def high_freq_ohe(train, test, extra_cols, target, n_limit=50):
    """
    對高基數類別特徵進行處理：保留頻率最高的 n_limit 個類別，其餘合併，再 OHE。
    (保持原邏輯不變)
    """
    train_copy = train.copy()
    test_copy = test.copy()
    print(f"對高基數特徵 {extra_cols} 進行頻率限制 (top {n_limit}) 並 OHE...")
    for col in extra_cols:
        if col not in train_copy.columns: continue
        dict1 = train_copy[col].value_counts().to_dict()
        ordered = dict(sorted(dict1.items(), key=lambda x: x[1], reverse=True))
        rare_keys = list([*ordered.keys()][n_limit:])
        rare_key_map = dict(zip(rare_keys, np.full(len(rare_keys), 9999)))
        train_copy[col] = train_copy[col].replace(rare_key_map)
        if col in test_copy.columns: # 確保測試集也有此欄位
             test_copy[col] = test_copy[col].replace(rare_key_map)
    train_copy, test_copy = OHE(train_copy, test_copy, extra_cols, target)
    drop_cols = [f for f in train_copy.columns if "_OHE_9999" in f or train_copy[f].nunique() == 1]
    train_copy = train_copy.drop(columns=drop_cols, errors='ignore')
    test_copy = test_copy.drop(columns=[col for col in drop_cols if col in test_copy.columns], errors='ignore')
    return train_copy, test_copy

def cat_encoding(train, test, cat_cols_input, target):
    """
    對類別特徵進行多種編碼 (Count, Count+Label, OHE/HighFreqOHE)，
    並使用單變數模型評估效果，保留最佳編碼方式產生的特徵，
    同時移除與最佳特徵高度相關的其他編碼特徵。
    Args:
        train (pd.DataFrame): 訓練集。
        test (pd.DataFrame): 測試集。
        cat_cols_input (list): 要進行編碼的類別欄位列表 (帶 'cat_' 前綴)。
        target (str): 目標變數名稱。
    Returns:
        tuple: 編碼後的 (訓練集, 測試集)。
    """
    global overall_best_score
    global overall_best_col
    table = PrettyTable() # 用於顯示編碼結果
    table.field_names = ['原始特徵', '最佳編碼特徵', 'ROC AUC 分數']
    train_copy = train.copy()
    test_copy = test.copy()
    print("開始應用並評估類別特徵編碼...")

    for feature in tqdm(cat_cols_input, desc="處理類別特徵編碼"):
        temp_cols = [] # 儲存本次迴圈產生的所有編碼欄位名稱

        # Count Encoding (頻率編碼)
        try:
            dic_count = train_copy[feature].value_counts().to_dict()
            train_copy[feature + "_count"] = train_copy[feature].map(dic_count)
            test_copy[feature + "_count"] = test_copy[feature].map(dic_count)
            # 對測試集中可能出現的新類別，用 0 或其他策略填充 NaN
            test_copy[feature + "_count"] = test_copy[feature + "_count"].fillna(0)
            temp_cols.append(feature + "_count")
        except Exception as e:
             print(f"警告：對欄位 {feature} 進行 Count Encoding 失敗 - {e}")

        # Count + Label Encoding (基於頻率排名的標籤編碼)
        try:
            dic2_rank = train_copy[feature].value_counts().to_dict()
            list1_rank = np.arange(len(dic2_rank.values()), 0, -1) # 頻率越高，標籤值越大
            dic3_rank = dict(zip(list(dic2_rank.keys()), list1_rank))
            train_copy[feature + "_count_label"] = train_copy[feature].replace(dic3_rank).astype(float)
            test_copy[feature + "_count_label"] = test_copy[feature].replace(dic3_rank).astype(float)
            # 填充測試集中的新類別 (給予最低排名 0)
            test_copy[feature + "_count_label"] = test_copy[feature + "_count_label"].fillna(0)
            temp_cols.append(feature + "_count_label")
        except Exception as e:
             print(f"警告：對欄位 {feature} 進行 Count+Label Encoding 失敗 - {e}")


        # OHE 或 High Frequency OHE
        try:
            if train_copy[feature].dtype == 'O' or train_copy[feature].nunique() <= 50 : # 如果是字串或低基數
                if train_copy[feature].nunique() > 2: # 避免對二元特徵再做OHE
                    # 如果原本不是字串，先轉成字串再 OHE，以防數值本身被誤解
                    if train_copy[feature].dtype != 'O':
                         train_copy[feature] = train_copy[feature].astype(str) + "_" + feature # 添加後綴區分
                         test_copy[feature] = test_copy[feature].astype(str) + "_" + feature
                    train_copy, test_copy = OHE(train_copy, test_copy, [feature], target) # OHE內部會移除原欄位
                # else: # 如果是二元特徵，保留原始欄位（通常已是 0/1）
                #      temp_cols.append(feature)
            else: # 高基數類別特徵 (> 50)
                train_copy, test_copy = high_freq_ohe(train_copy, test_copy, [feature], target, n_limit=10) # 保留 top 10，其餘合併 OHE
                # high_freq_ohe 內部會移除原欄位
        except Exception as e:
             print(f"警告：對欄位 {feature} 進行 OHE/HighFreqOHE 失敗 - {e}")

        # 將 OHE 產生的新欄位也加入評估列表
        ohe_generated_cols = [c for c in train_copy.columns if feature + "_OHE_" in c]
        temp_cols.extend(ohe_generated_cols)

        # 填補因編碼可能產生的缺失值 (例如 test set 的 map)
        train_copy, test_copy = fill_missing_numerical(train_copy, test_copy, target, 3)

        # 使用交叉驗證評估每個編碼的效果
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = []
        eval_cols = [c for c in temp_cols if c in train_copy.columns] # 確保欄位存在

        for f in eval_cols:
            if train_copy[f].isnull().any(): # 跳過仍有缺失值的欄位
                print(f"警告：欄位 {f} 在評估前仍有缺失值，跳過評估。")
                continue

            X = train_copy[[f]].values
            y = train_copy[target].astype(int).values

            auc = []
            try:
                for train_idx, val_idx in kf.split(X, y):
                    X_train_eval, y_train_eval = X[train_idx], y[train_idx]
                    x_val_eval, y_val_eval = X[val_idx], y[val_idx]
                    # 使用 HistGradientBoostingClassifier 進行快速評估
                    model_eval = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.02, max_depth=6, random_state=42)
                    model_eval.fit(X_train_eval, y_train_eval)
                    y_pred_eval = model_eval.predict_proba(x_val_eval)[:, 1]
                    auc.append(roc_auc_score(y_val_eval, y_pred_eval))
                mean_auc = np.mean(auc)
                auc_scores.append((f, mean_auc))
                # 更新全局最佳分數
                if overall_best_score < mean_auc:
                    overall_best_score = mean_auc
                    overall_best_col = f
            except Exception as e:
                 print(f"警告：評估欄位 {f} 時出錯 - {e}")

        if not auc_scores: # 如果沒有任何編碼成功或被評估
            print(f"警告：欄位 {feature} 的所有編碼均未成功評估。")
            # 考慮是否保留原始 'cat_' 特徵或移除
            if feature in train_copy.columns:
                 train_copy = train_copy.drop(columns=[feature])
            if feature in test_copy.columns:
                 test_copy = test_copy.drop(columns=[feature])
            continue

        # 找出效果最好的編碼
        best_col, best_auc = sorted(auc_scores, key=lambda x: x[1], reverse=True)[0]

        # 計算其他編碼與最佳編碼的相關性
        corr_eval_cols = [c for c in eval_cols if c in train_copy.columns and c != best_col] # 確保欄位存在且非最佳欄位
        if corr_eval_cols: # 如果有其他欄位可以計算相關性
            try:
                corr = train_copy[[best_col] + corr_eval_cols].corr(method='pearson')
                corr_with_best_col = corr[best_col].drop(best_col) # 移除自身相關性
                # 找出與最佳編碼高度相關 (Corr > 0.5) 的其他編碼
                cols_to_drop = [f for f in corr_eval_cols if abs(corr_with_best_col.get(f, 0)) > 0.5] # 使用 .get 避免 KeyEroor
            except Exception as e:
                print(f"警告：計算欄位 {feature} 編碼相關性時出錯 - {e}")
                cols_to_drop = [f for f in corr_eval_cols] # 若計算失敗，保守起見移除所有非最佳
        else:
             cols_to_drop = []


        # final_selection = [f for f in eval_cols if f not in cols_to_drop] # 保留的欄位
        if cols_to_drop:
            train_copy = train_copy.drop(columns=[f for f in cols_to_drop if f in train_copy.columns]) # 從訓練集移除
            test_copy = test_copy.drop(columns=[f for f in cols_to_drop if f in test_copy.columns])   # 從測試集移除

        table.add_row([feature, best_col, f"{best_auc:.4f}"]) # 記錄結果
        # print(feature) # 顯示當前處理的特徵 (已被 tqdm 取代)

    print("\n類別特徵編碼評估結果:")
    print(table)
    print(f"所有類別編碼中最佳的單變數 CV ROC AUC 分數: {overall_best_score:.4f} (來自特徵: {overall_best_col})")
    # 再次填補，以防移除欄位後產生新的 NaN（雖然理論上不太可能）
    train_copy, test_copy = fill_missing_numerical(train_copy, test_copy, target, 3)
    return train_copy, test_copy

def better_features(train, test, target, cols, best_score):
    """
    自動搜索最佳的算術組合特徵 (+, -, *, /)。
    (保持原邏輯不變)
    """
    new_cols = []
    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    print("開始自動搜索算術組合特徵 (計算量可能很大)...")
    for i in tqdm(range(len(cols)), desc='生成組合特徵'):
        col1 = cols[i]
        temp_df = pd.DataFrame(index=train.index)
        temp_df_test = pd.DataFrame(index=test.index)

        for j in range(i + 1, len(cols)):
            col2 = cols[j]
            # (乘法, 除法, 減法, 加法 邏輯不變)
            # ...
            temp_df[f'{col1}*{col2}'] = train[col1] * train[col2]
            temp_df_test[f'{col1}*{col2}'] = test[col1] * test[col2]
            temp_df[f'{col1}/{col2}'] = train[col1] / (train[col2] + 1e-9)
            temp_df_test[f'{col1}/{col2}'] = test[col1] / (test[col2] + 1e-9)
            temp_df[f'{col2}/{col1}'] = train[col2] / (train[col1] + 1e-9)
            temp_df_test[f'{col2}/{col1}'] = test[col2] / (test[col1] + 1e-9)
            temp_df[f'{col1}-{col2}'] = train[col1] - train[col2]
            temp_df_test[f'{col1}-{col2}'] = test[col1] - test[col2]
            temp_df[f'{col1}+{col2}'] = train[col1] + train[col2]
            temp_df_test[f'{col1}+{col2}'] = test[col1] + test[col2]


        SCORES = []
        for column in temp_df.columns:
            if temp_df[column].isnull().any(): continue
            scores = []
            try:
                for train_index, val_index in skf.split(train, train[target]):
                    X_train_eval = temp_df[column].iloc[train_index].values.reshape(-1, 1)
                    X_val_eval = temp_df[column].iloc[val_index].values.reshape(-1, 1)
                    y_train_eval = train[target].astype(int).iloc[train_index]
                    y_val_eval = train[target].astype(int).iloc[val_index]
                    model_eval = LogisticRegression(solver='liblinear')
                    model_eval.fit(X_train_eval, y_train_eval)
                    y_pred_eval = model_eval.predict_proba(X_val_eval)[:, 1]
                    score = roc_auc_score(y_val_eval, y_pred_eval)
                    scores.append(score)
                mean_score = np.mean(scores)
                SCORES.append((column, mean_score))
            except Exception as e: print(f"警告：評估組合特徵 {column} 時出錯 - {e}")

        if SCORES:
            best_col, best_auc = sorted(SCORES, key=lambda x: x[1], reverse=True)[0]
            try:
                corr_with_other_cols = train.drop([target] + new_cols, axis=1, errors='ignore').corrwith(temp_df[best_col])
                max_corr = corr_with_other_cols.abs().max() if not corr_with_other_cols.empty else 0
                if (max_corr < 0.9 or best_auc > best_score) and not np.isclose(max_corr, 1.0): # 修正: 處理浮點數比較
                    train[best_col] = temp_df[best_col]
                    test[best_col] = temp_df_test[best_col]
                    new_cols.append(best_col)
                    print(f"已加入組合特徵 '{best_col}' (ROC AUC: {best_auc:.4f}, Max Corr: {max_corr:.4f})")
            except Exception as e: print(f"警告：計算特徵 {best_col} 的相關性或加入時出錯 - {e}")

    print("算術組合特徵搜索完成。")
    # 確保測試集有訓練集的所有欄位
    missing_cols_in_test_arith = list(set(train.columns) - set(test.columns) - set([target]))
    for c in missing_cols_in_test_arith: test[c] = 0 # 或其他填充策略
    if target in train.columns:
         test = test[train.drop(columns=[target]).columns]
    else:
         test = test[train.columns]

    return train, test, new_cols

def post_processor(train_df, test_df):
    """
    比較所有非 OHE 特徵對，移除完全相同的重複特徵。
    (保持原邏輯不變)
    """
    print("檢查並移除完全相同的特徵...")
    cols_to_compare = [f for f in train_df.columns if "smoking" not in f and "OHE" not in f and f in test_df.columns]
    train_cop = train_df.copy()
    test_cop = test_df.copy()
    drop_cols = []
    for i in tqdm(range(len(cols_to_compare)), desc="比較特徵對"):
        feature = cols_to_compare[i]
        if feature in drop_cols: continue
        for j in range(i + 1, len(cols_to_compare)):
            compare_feature = cols_to_compare[j]
            if compare_feature in drop_cols: continue
            try:
                 if np.allclose(train_cop[feature].fillna(0), train_cop[compare_feature].fillna(0), atol=1e-8, equal_nan=True): # 填充NaN再比較
                    if compare_feature not in drop_cols: drop_cols.append(compare_feature)
            except Exception as e: print(f"警告：比較特徵 {feature} 和 {compare_feature} 時出錯 - {e}")

    print(f"找到以下完全相同的重複特徵將被移除: {drop_cols}")
    train_cop.drop(columns=drop_cols, inplace=True, errors='ignore')
    test_cop.drop(columns=drop_cols, inplace=True, errors='ignore')
    print("重複特徵移除完成。")
    return train_cop, test_cop

def get_most_important_features(X_train_fs, y_train_fs, n, model_input):
    """
    使用指定模型計算特徵重要性並返回 top n。
    (修正 LGBM fit 調用)
    """
    print(f"\n使用 {model_input.upper()} 計算特徵重要性 (選擇 top {n})...")
    xgb_params_fs = {'n_jobs': -1, 'eval_metric': 'logloss', 'objective': 'binary:logistic', 'tree_method': 'hist', 'verbosity': 0, 'random_state': 42}
    if device == 'gpu': xgb_params_fs['tree_method'], xgb_params_fs['predictor'] = 'gpu_hist', 'gpu_predictor'
    lgb_params_fs = {'objective': 'binary', 'metric': 'logloss', 'boosting_type': 'gbdt', 'random_state': 42, 'device': device, 'verbosity': -1}
    cb_params_fs = {'iterations': 500, 'grow_policy': 'Depthwise', 'bootstrap_type': 'Bayesian', 'od_type': 'Iter', 'od_wait': 50, 'eval_metric': 'AUC', 'loss_function': 'Logloss', 'random_state': 42, 'task_type': device.upper(), 'verbose': 0}

    if 'xgb' in model_input: model_fs = xgb.XGBClassifier(**xgb_params_fs)
    elif 'cat' in model_input: model_fs = CatBoostClassifier(**cb_params_fs)
    else: model_fs = lgb.LGBMClassifier(**lgb_params_fs)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    feature_importances_list = []

    for fold, (train_idx, val_idx) in enumerate(tqdm(kfold.split(X_train_fs, y_train_fs), total=5, desc=f"計算 {model_input.upper()} 重要性")):
        X_train_fold, X_val_fold = X_train_fs.iloc[train_idx], X_train_fs.iloc[val_idx]
        y_train_fold, y_val_fold = y_train_fs.iloc[train_idx], y_train_fs.iloc[val_idx]

        # --- 修正：LGBM 的 fit 調用 ---
        try:
            if 'cat' in model_input:
                 model_fs.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], early_stopping_rounds=50, verbose=0)
            elif 'lgb' in model_input:
                 # 在此函數中不使用早停或 eval_set，避免之前的錯誤
                 model_fs.fit(X_train_fold, y_train_fold, verbose=0)
            else: # xgb
                 model_fs.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=0) # XGBoost >= 0.17
        except Exception as e:
            print(f"錯誤：模型 {model_input.upper()} 在 Fold {fold+1} 訓練失敗 - {e}")
            continue # 跳過此折
        # --- 修正結束 ---

        try:
            y_pred_fs = model_fs.predict_proba(X_val_fold)[:, 1]
            auc_scores.append(roc_auc_score(y_val_fold, y_pred_fs))
            if hasattr(model_fs, 'feature_importances_'):
                 feature_importances_list.append(model_fs.feature_importances_)
            else: # CatBoost 可能沒有這個屬性，需要用 get_feature_importance()
                 try:
                     feature_importances_list.append(model_fs.get_feature_importance())
                 except: # 如果兩者都沒有，填充 0
                      feature_importances_list.append(np.zeros(X_train_fs.shape[1]))

        except Exception as e: print(f"警告：在 fold {fold+1} 計算 {model_input.upper()} 重要性時出錯 - {e}")

    if not feature_importances_list: print(f"錯誤：無法計算 {model_input.upper()} 的特徵重要性。"); return list(X_train_fs.columns[:n])

    # 確保所有重要性列表長度一致
    min_len = min(len(fi) for fi in feature_importances_list if hasattr(fi, '__len__'))
    valid_importances = [fi[:min_len] for fi in feature_importances_list if hasattr(fi, '__len__') and len(fi) >= min_len]
    if not valid_importances: print(f"錯誤：特徵重要性列表無效。"); return list(X_train_fs.columns[:n])

    avg_feature_importances = np.mean(valid_importances, axis=0)
    feature_importance_list = [(X_train_fs.columns[i], importance) for i, importance in enumerate(avg_feature_importances)]
    sorted_features = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)
    top_n_features = [feature[0] for feature in sorted_features[:n]]
    return top_n_features


class Splitter:
    """用於數據分割的類別，支持 KFold 交叉驗證。"""
    def __init__(self, kfold=True, n_splits=5):
        self.kfold = kfold
        self.n_splits = n_splits

    def split_data(self, X, y, random_state_list):
        """
        產生交叉驗證的索引。
        Yields:
            tuple: (fold_index, train_index, val_index, current_seed, current_fold_num)
        """
        fold_counter = 0
        for seed in random_state_list:
            if self.kfold:
                kf = KFold(n_splits=self.n_splits, random_state=seed, shuffle=True)
                print(f"\n使用 KFold (n_splits={self.n_splits}, random_state={seed}) 分割數據...")
                for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
                    yield fold_counter, train_index, val_index, seed, fold
                    fold_counter += 1
            else: # 簡單訓練一次（非交叉驗證）
                 print("警告：非 KFold 模式，將使用全部數據訓練一次。")
                 yield 0, X.index, X.index, seed, 0 # 返回完整索引
                 break # 只執行一次

class Classifier:
    """包含多種分類模型及其參數設定的類別。"""
    def __init__(self, n_estimators=100, device="gpu", random_state=0):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.models = self._define_model()
        self.len_models = len(self.models)

    def _define_model(self):
        """定義模型字典。"""
        print("定義多個分類模型及其參數...")
        # (模型參數定義與之前相同，確保 'ann' 已根據需要移除)
        xgb_params = {'n_estimators': self.n_estimators, 'learning_rate': 0.155, 'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 0.1, 'n_jobs': -1, 'eval_metric': 'logloss', 'objective': 'binary:logistic', 'tree_method': 'hist', 'verbosity': 0, 'random_state': self.random_state}
        if self.device == 'gpu': xgb_params['tree_method'], xgb_params['predictor'] = 'gpu_hist', 'gpu_predictor'
        xgb_params2 = xgb_params.copy(); xgb_params2.update({'subsample': 0.3, 'max_depth': 8, 'learning_rate': 0.005, 'colsample_bytree': 0.9})
        xgb_params3 = xgb_params.copy(); xgb_params3.update({'subsample': 0.6, 'max_depth': 6, 'learning_rate': 0.0125, 'colsample_bytree': 0.7})
        lgb_params = {'n_estimators': self.n_estimators, 'max_depth': 8, 'learning_rate': 0.0125, 'subsample': 0.20, 'colsample_bytree': 0.56, 'reg_alpha': 0.25, 'reg_lambda': 5e-08, 'objective': 'binary', 'boosting_type': 'gbdt', 'device': self.device, 'random_state': self.random_state, 'verbosity': -1}
        lgb_params2 = lgb_params.copy(); lgb_params2.update({'max_depth': 6, 'learning_rate': 0.0375})
        lgb_params3 = lgb_params.copy(); lgb_params3.update({'subsample': 0.9, 'reg_lambda': 0.346, 'reg_alpha': 0.310, 'max_depth': 8, 'learning_rate': 0.0075, 'colsample_bytree': 0.5})
        cb_params = {'iterations': self.n_estimators, 'depth': 6, 'learning_rate' : 0.0075, 'l2_leaf_reg': 0.7, 'random_strength': 0.2, 'max_bin': 200, 'od_wait': 65, 'one_hot_max_size': 120, 'grow_policy': 'Depthwise', 'bootstrap_type': 'Bayesian', 'od_type': 'Iter', 'eval_metric': 'AUC', 'loss_function': 'Logloss', 'task_type': self.device.upper(), 'random_state': self.random_state, 'verbose': 0}
        cb_sym_params = cb_params.copy(); cb_sym_params['grow_policy'] = 'SymmetricTree'
        cb_loss_params = cb_params.copy(); cb_loss_params['grow_policy'] = 'Lossguide'
        cb_params2 = cb_params.copy(); cb_params2.update({'learning_rate': 0.075, 'depth': 8})
        cb_params3 = cb_params2.copy()
        dt_params = {'min_samples_split': 30, 'min_samples_leaf': 10, 'max_depth': 8, 'criterion': 'gini'}
        models = {
            'xgb': xgb.XGBClassifier(**xgb_params), 'xgb2': xgb.XGBClassifier(**xgb_params2), 'xgb3': xgb.XGBClassifier(**xgb_params3),
            'lgb': lgb.LGBMClassifier(**lgb_params), 'lgb2': lgb.LGBMClassifier(**lgb_params2), 'lgb3': lgb.LGBMClassifier(**lgb_params3),
            'cat': CatBoostClassifier(**cb_params), 'cat2': CatBoostClassifier(**cb_params2), 'cat3': CatBoostClassifier(**cb_params3),
            "cat_sym": CatBoostClassifier(**cb_sym_params), "cat_loss": CatBoostClassifier(**cb_loss_params),
            'hist_gbm': HistGradientBoostingClassifier(max_iter=300, learning_rate=0.001, max_leaf_nodes=80, max_depth=6, random_state=self.random_state),
            'lr': LogisticRegression(solver='liblinear', random_state=self.random_state),
            'dt': DecisionTreeClassifier(**dt_params, random_state=self.random_state)
        }
        print(f"已定義 {len(models)} 個基礎模型。")
        return models

# ==============================================================================
# ======================== 主要執行邏輯開始 ==============================
# ==============================================================================

if __name__ == "__main__": # 將主邏輯放入 main block

    if TRAIN_MODE:
        # ========================== 訓練模式 ==========================
        print("===== 模式：訓練 & 儲存 =====")

        # --- 1. 載入訓練數據 ---
        print("正在讀取資料...")
        try:
            train = pd.read_csv(TRAIN_FILE)
            # --- 載入 original/extend data ---
            original = pd.read_csv(ORIGINAL_FILE)
            # --- 載入 test data (for training mode) ---
            test = pd.read_csv(TEST_FILE_FOR_TRAINING)
            test_ids = test['id'].copy() # 保存測試集 ID
        except FileNotFoundError as e:
            print(f"錯誤：找不到訓練所需檔案 {e.filename}。請檢查路徑設定。")
            exit()
        except Exception as e:
            print(f"讀取數據時發生錯誤: {e}")
            exit()

        # --- >>> 新增：載入篩選索引並過濾擴增數據 <<< ---
        print(f"\n嘗試從 {INDICES_FILE_PATH} 載入篩選後的擴增數據索引...")
        selected_indices = None
        original_filtered = original.copy() # 預設使用完整數據

        if os.path.exists(INDICES_FILE_PATH):
            try:
                with open(INDICES_FILE_PATH, 'rb') as f:
                    selected_indices = pickle.load(f)

                # 檢查載入的索引是否有效 (例如是否為 pandas Index 或類似列表的結構)
                if hasattr(selected_indices, '__len__') and len(selected_indices) > 0:
                    # 確保索引存在於 original DataFrame 中
                    valid_indices = original.index.intersection(selected_indices)
                    if len(valid_indices) != len(selected_indices):
                        print(f"警告：載入的索引中有 {len(selected_indices) - len(valid_indices)} 個在原始擴增數據中不存在，將只使用有效索引。")
                    if len(valid_indices) > 0:
                        original_filtered = original.loc[valid_indices].copy() # 使用 .loc 和有效索引過濾
                        print(f"成功載入並應用索引。擴增數據已篩選，保留 {original_filtered.shape[0]} / {original.shape[0]} 筆樣本。")
                    else:
                        print(f"警告：載入的索引與當前擴增數據完全不匹配。將使用完整的擴增數據集。")
                else:
                    print(f"警告：載入的索引檔案 '{INDICES_FILE_PATH}' 為空或格式不正確。將使用完整的擴增數據集。")

            except Exception as e:
                print(f"錯誤：載入或應用索引檔案 {INDICES_FILE_PATH} 失敗 - {e}")
                print("將使用完整的擴增數據集。")
        else:
            print(f"警告：找不到索引檔案 {INDICES_FILE_PATH}。這可能是您第一次運行或未執行鑑別器腳本。將使用完整的擴增數據集。")
        # --- >>> 新增結束 <<< ---


        # --- 2. 資料預處理與合併 ---
        print("\n開始預處理與合併數據...")
        try:
            train_id_exists = 'id' in train.columns
            test_id_exists = 'id' in test.columns
            if train_id_exists: train.drop(columns=["id"], inplace=True)
            if test_id_exists: test.drop(columns=["id"], inplace=True)

            # --- 修改合併邏輯 ---
            original_filtered["original"] = 1 # <--- 對篩選後的數據添加來源標籤
            train["original"] = 0
            test["original"] = 0

            # 使用篩選後的 original_filtered 進行合併
            train = pd.concat([train, original_filtered], axis=0) # <--- 使用 original_filtered
            train.reset_index(inplace=True, drop=True)
            # --- 修改結束 ---

            print("資料讀取、篩選與初步合併完成。")
            print(f"合併後訓練資料維度: {train.shape}") # <-- 打印合併後的維度

        except Exception as e:
            print(f"預處理與合併數據時發生錯誤: {e}")
            exit()

        # --- 3. 特徵工程 (對 train 和 test 同步進行) ---
        print("\n開始特徵工程...")
        try:
            train = create_extra_features(train.copy()) # 使用 .copy() 避免 SettingWithCopyWarning
            test = create_extra_features(test.copy())

            print("\n開始數值特徵轉換...")
            # 找出數值型且唯一值數量 > 50 的欄位進行轉換
            cont_cols = [f for f in train.columns if pd.api.types.is_numeric_dtype(train[f]) and train[f].nunique() > 50 and f != 'original'] # 排除 original 標記欄位
            # 找出唯一值數量介於 2 和 50 之間，且不是目標 'smoking' 的欄位，作為離散特徵處理
            cat_cols = [f for f in train.columns if train[f].nunique() > 2 and train[f].nunique() <= 50 and f not in ['smoking', 'original']]

            print("剩餘訓練集維度:", train.shape)
            print(f"待進行轉換的連續數值特徵 ({len(cont_cols)}): {cont_cols}")
            print(f"待作為類別處理的離散數值特徵 ({len(cat_cols)}): {cat_cols}")

            # sc=MinMaxScaler() # 不在此處初始化，在 min_max_scaler 函數內處理

            global unimportant_features # 儲存效果不佳的轉換後特徵
            global overall_best_score # 記錄所有轉換中的最佳單變數模型分數
            global overall_best_col # 記錄產生最佳分數的特徵名稱
            unimportant_features = []
            overall_best_score = 0
            overall_best_col = 'none'

            # 對連續數值欄位進行 Min-Max 標準化
            print("對連續數值特徵進行 Min-Max 標準化...")
            for col in cont_cols:
                train, test = min_max_scaler(train, test, col)
            print("Min-Max 標準化完成。")

            # 執行數值轉換
            train, test = transformer(train, test, cont_cols, "smoking")
            print("數值特徵轉換完成。")
            print("剩餘訓練集維度:", train.shape)

            print("\n開始處理離散數值特徵...")
            selected_cols = []
            for col in cat_cols:
                 if col in train.columns: # 確保欄位存在
                     train['cat_' + col] = train[col]
                     if col in test.columns: test['cat_' + col] = test[col]
                     else: test['cat_' + col] = 0 # 如果測試集沒有此欄，填充 0
                     selected_cols.append('cat_' + col)
            train, test = cat_encoding(train, test, selected_cols, "smoking")
            print("類別特徵編碼完成。")
            print("剩餘訓練集維度:", train.shape)

            if ENABLE_BETTER_FEATURES:
                print("啟用動態算術特徵搜索 (better_features)...")
                cols_for_arithmetic = [f for f in train.columns if pd.api.types.is_numeric_dtype(train[f]) and 'OHE' not in f and f not in ['smoking', 'original'] and train[f].nunique() > 1 and f in test.columns] # 確保測試集也有
                print(f"將使用 {len(cols_for_arithmetic)} 個特徵進行組合搜索...")
                train, test, generated_new_cols = better_features(train, test, 'smoking', cols_for_arithmetic, overall_best_score)
                print(f"動態算術特徵搜索完成，新增了 {len(generated_new_cols)} 個特徵。")
            else:
                print("警告：已禁用動態算術搜索。")

            first_drop = [f for f in unimportant_features if f in train.columns]
            train = train.drop(columns=first_drop, errors='ignore')
            test = test.drop(columns=first_drop, errors='ignore')
            print("特徵工程完成。")
            print("特徵工程後 Train shape:", train.shape) # <--- 新增打印
        except Exception as e:
             print(f"特徵工程階段發生錯誤: {e}")
             import traceback
             traceback.print_exc() # 打印詳細錯誤追蹤
             exit()


        # --- 4. 特徵選擇 (包含儲存 Scaler 和特徵列表) ---
        print("\n開始特徵選擇...")
        try:
            target_col = 'smoking'
            features_before_selection = [f for f in train.columns if f != target_col]
            missing_cols_final = list(set(features_before_selection) - set(test.columns))
            for col in missing_cols_final: test[col] = 0
            test = test[features_before_selection]
            print(f"特徵選擇前數量: {len(features_before_selection)}")

            sc = StandardScaler()
            train_scaled = train.copy()
            test_scaled = test.copy()
            features_to_scale = features_before_selection
            fill_values = train_scaled[features_to_scale].mean()
            train_scaled[features_to_scale] = train_scaled[features_to_scale].fillna(fill_values)
            test_scaled[features_to_scale] = test_scaled[features_to_scale].fillna(fill_values)
            train_scaled[features_to_scale] = sc.fit_transform(train_scaled[features_to_scale])
            test_scaled[features_to_scale] = sc.transform(test_scaled[features_to_scale])
            print("標準化完成。")
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            joblib.dump(sc, SCALER_FILE); print(f"StandardScaler 已儲存到: {SCALER_FILE}")

            train_cop, test_cop = post_processor(train_scaled, test_scaled)
            X_train_processed = train_cop.drop(columns=[target_col], errors='ignore')
            y_train_final = train[target_col].astype(int)
            X_test_processed = test_cop.copy()
            common_cols_processed = [col for col in X_train_processed.columns if col in X_test_processed.columns]
            X_train_processed = X_train_processed[common_cols_processed]
            X_test_processed = X_test_processed[common_cols_processed]

            # --- 新增：選擇並儲存 Top N 原始特徵 ---
            print(f"\n選擇 Top {TOP_N_ORIGINAL_FEATURES} 個原始特徵用於元模型...")
            try:
                # 使用快速 LGBM 評估 X_train_processed 的特徵重要性
                lgbm_feat_selector = lgb.LGBMClassifier(random_state=42, n_estimators=100, verbose=-1)
                lgbm_feat_selector.fit(X_train_processed, y_train_final)
                importances = lgbm_feat_selector.feature_importances_
                feat_importance_df = pd.DataFrame({
                    'feature': X_train_processed.columns,
                    'importance': importances
                })
                # 篩選看起來像「原始」的特徵（ heuristic ）
                original_like_features = feat_importance_df[
                    ~feat_importance_df['feature'].str.contains('_OHE_|_count|_label|_pca_comb|\*|/|\+|-|\^cat_', regex=True, na=False)
                ]
                top_original_features = original_like_features.sort_values(
                    'importance', ascending=False
                ).head(TOP_N_ORIGINAL_FEATURES)['feature'].tolist()

                if len(top_original_features) < TOP_N_ORIGINAL_FEATURES:
                     print(f"警告：只找到 {len(top_original_features)} 個原始特徵，少於要求的 {TOP_N_ORIGINAL_FEATURES} 個。")
                if not top_original_features: # 如果一個都沒找到
                     print("錯誤：未能選出任何原始特徵！請檢查篩選邏輯。")
                     # 可以選擇退出或不加入原始特徵
                     top_original_features = [] # 設為空列表

                print(f"選出的 Top {len(top_original_features)} 原始特徵: {top_original_features}")
                # 儲存列表
                joblib.dump(top_original_features, TOP_FEATURES_FILE)
                print(f"Top {len(top_original_features)} 原始特徵列表已儲存到: {TOP_FEATURES_FILE}")
            except Exception as e_feat:
                print(f"錯誤：選擇 Top {TOP_N_ORIGINAL_FEATURES} 原始特徵失敗 - {e_feat}")
                top_original_features = [] # 出錯則不使用原始特徵
            # --- 新增結束 ---


            # 基於模型進行最終特徵選擇 (選擇用於 Level 0 模型的特徵)
            n_imp_features_cat = get_most_important_features(X_train_processed.reset_index(drop=True), y_train_final, N_FEATURES_TO_SELECT, 'cat')
            n_imp_features_xgb = get_most_important_features(X_train_processed.reset_index(drop=True), y_train_final, N_FEATURES_TO_SELECT, 'xgb')
            n_imp_features_lgbm = get_most_important_features(X_train_processed.reset_index(drop=True), y_train_final, N_FEATURES_TO_SELECT, 'lgb')
            n_imp_features_final = [*set(n_imp_features_xgb + n_imp_features_lgbm)]
            print(f"\n最終選擇了 {len(n_imp_features_final)} 個特徵進行 Level 0 模型訓練。")
            joblib.dump(n_imp_features_final, FINAL_FEATURES_FILE); print(f"最終特徵列表已儲存到: {FINAL_FEATURES_FILE}")

            X_train_selected = X_train_processed[n_imp_features_final]
            X_test_selected = X_test_processed[n_imp_features_final]

            print("特徵選擇完成。")
            print("Level 0 模型輸入維度 (訓練):", X_train_selected.shape)
            print("Level 0 模型輸入維度 (測試):", X_test_selected.shape)

        except Exception as e: print(f"特徵選擇階段發生錯誤: {e}"); import traceback; traceback.print_exc(); exit()


        # --- 5. 模型訓練與 Stacking (Level 0 訓練與儲存) ---
        print("\n開始 Level 0 模型訓練與 Stacking...")
        try:
            splitter = Splitter(kfold=True, n_splits=N_SPLITS)
            temp_classifier = Classifier(n_estimators=10, device=device, random_state=RANDOM_STATE_LIST[0])
            model_names = list(temp_classifier.models.keys())
            n_models = len(model_names)
            del temp_classifier

            oof_predictions = np.zeros((len(X_train_selected), n_models))
            test_predictions_avg = np.zeros((len(X_test_selected), n_models))
            oof_scores = {}
            total_folds = len(RANDOM_STATE_LIST) * N_SPLITS

            for fold_i, train_index, val_index, current_random_state, n in splitter.split_data(X_train_selected, y_train_final, random_state_list=RANDOM_STATE_LIST):
                X_train_, X_val = X_train_selected.iloc[train_index], X_train_selected.iloc[val_index]
                y_train_, y_val = y_train_final.iloc[train_index], y_train_final.iloc[val_index]

                print(f"\n--- 開始訓練 Level 0: Fold {n+1}/{N_SPLITS}, Seed {current_random_state} ({fold_i+1}/{total_folds}) ---")
                classifier = Classifier(N_ESTIMATORS, device, current_random_state)
                models = classifier.models

                for model_idx, (name, model) in enumerate(tqdm(models.items(), desc=f"訓練 Fold-{n+1} Seed-{current_random_state}")):
                    try:
                        # 訓練模型
                        if ('cat' in name) or ("lgb" in name) or ("xgb" in name):
                            if 'lgb' in name: model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], callbacks=[lgb_callback.early_stopping(EARLY_STOPPING_ROUNDS, verbose=verbose)])
                            elif 'cat' in name: model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=verbose)
                            else: model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], verbose=verbose)
                        else: model.fit(X_train_, y_train_)

                        # 儲存模型
                        try:
                            model_filename = os.path.join(MODEL_SAVE_DIR, f'model_{name}_fold{n+1}_seed{current_random_state}.pkl')
                            joblib.dump(model, model_filename)
                        except Exception as e: print(f"    警告：儲存模型 {name} (Fold {n+1}, Seed {current_random_state}) 失敗 - {e}")

                        # 預測
                        if hasattr(model, 'predict_proba'):
                             test_pred = model.predict_proba(X_test_selected)[:, 1]; y_val_pred = model.predict_proba(X_val)[:, 1]
                        else: test_pred = model.predict(X_test_selected).astype(float); y_val_pred = model.predict(X_val).astype(float)

                        score = roc_auc_score(y_val, y_val_pred)
                        print(f'  {name} [Fold-{n+1} Seed-{current_random_state}] OOF ROC AUC: {score:.5f}')
                        oof_predictions[val_index, model_idx] = y_val_pred
                        test_predictions_avg[:, model_idx] += test_pred / total_folds
                        if name not in oof_scores: oof_scores[name] = []
                        oof_scores[name].append(score)

                    except Exception as e:
                        print(f"錯誤：訓練或預測模型 '{name}' (Fold {n+1}, Seed {current_random_state}) 時出錯 - {e}")
                        oof_predictions[val_index, model_idx] = np.nan
                        test_predictions_avg[:, model_idx] += np.full(len(X_test_selected), np.nan) / total_folds
                gc.collect()

            print("\n--- Level 0 模型交叉驗證訓練與儲存完成 ---")

            # NaN 處理
            if np.isnan(oof_predictions).any():
                 print("警告：OOF 預測中包含 NaN，將使用列平均值進行填補...")
                 col_means = np.nanmean(oof_predictions, axis=0)
                 inds = np.where(np.isnan(oof_predictions)); oof_predictions[inds] = np.take(col_means, inds[1])
                 if np.isnan(oof_predictions).any(): oof_predictions = np.nan_to_num(oof_predictions, nan=0.5)
            if np.isnan(test_predictions_avg).any():
                 print("警告：平均測試集預測中包含 NaN，將使用列平均值進行填補...")
                 col_means_test = np.nanmean(test_predictions_avg, axis=0)
                 inds_test = np.where(np.isnan(test_predictions_avg)); test_predictions_avg[inds_test] = np.take(col_means_test, inds_test[1])
                 if np.isnan(test_predictions_avg).any(): test_predictions_avg = np.nan_to_num(test_predictions_avg, nan=0.5)

            # --- 6. 訓練並儲存元模型 (Level 1) - 使用 ExtraTrees + Top 特徵 ---
            print("\n--- 開始訓練並儲存 Level 1 元模型 (使用 ExtraTrees + Top 特徵) ---")
            # 準備 Level 0 預測結果
            X_meta_train_preds = pd.DataFrame(oof_predictions, columns=model_names)
            X_meta_test_preds = pd.DataFrame(test_predictions_avg, columns=model_names)
            y_meta_train = y_train_final

            # --- 修改：合併 Top N 原始特徵 ---
            if top_original_features: # 只有在成功選出特徵時才合併
                 print(f"合併 Level 0 預測與 Top {len(top_original_features)} 原始特徵...")
                 # 從處理好但未經最終選擇的數據中選取 (X_train_processed, X_test_processed)
                 X_train_orig_feats = X_train_processed[top_original_features].reset_index(drop=True)
                 X_test_orig_feats = X_test_processed[top_original_features].reset_index(drop=True)

                 X_meta_train_final = pd.concat([X_meta_train_preds, X_train_orig_feats], axis=1)
                 X_meta_test_final = pd.concat([X_meta_test_preds, X_test_orig_feats], axis=1)
                 print(f"元模型最終輸入維度 (訓練): {X_meta_train_final.shape}")
                 print(f"元模型最終輸入維度 (測試): {X_meta_test_final.shape}")
            else: # 如果沒有選出原始特徵，只使用 Level 0 預測
                 print("警告：未使用 Top 原始特徵，元模型只基於 Level 0 預測。")
                 X_meta_train_final = X_meta_train_preds
                 X_meta_test_final = X_meta_test_preds
            # --- 修改結束 ---

            # 使用設定好的 ExtraTreesClassifier
            meta_model = META_MODEL_CLASS(**META_MODEL_PARAMS)
            print(f"--- 檢查：實際使用的元模型類型: {type(meta_model)} ---")

            # 訓練元模型 (ExtraTrees 不需要早停)
            meta_model.fit(X_meta_train_final, y_meta_train)
            print("元模型訓練完成。")

            # --- (儲存元模型邏輯不變) ---
            try:
                joblib.dump(meta_model, META_MODEL_FILE)
                print(f"元模型已儲存到: {META_MODEL_FILE}")
            except Exception as e:
                print(f"錯誤：儲存元模型失敗 - {e}")

            # 評估元模型 OOF 分數
            oof_stacking_score = roc_auc_score(y_meta_train, meta_model.predict_proba(X_meta_train_final)[:, 1])
            print(f'Stacking 元模型 OOF ROC AUC 分數: {oof_stacking_score:.5f}')

            test_predss = meta_model.predict_proba(X_meta_test_final)[:, 1]
            print("Stacking 最終預測產生完成。")

            # 產生提交檔案
            print("\n產生提交檔案 (針對訓練時使用的測試集)...")
            try:
                 sub = pd.DataFrame({'id': test_ids, 'smoking': test_predss})
                 sub.to_csv(OUTPUT_FILE_TRAINING_SUB, index=False)
                 print(f"已產生 {OUTPUT_FILE_TRAINING_SUB}")
                 print(sub.head())
            except NameError: print("錯誤: 找不到 test_ids。")
            except Exception as e: print(f"錯誤：產生提交檔案時出錯 - {e}")

        except Exception as e:
            print(f"模型訓練或 Stacking 階段發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            exit()

        print("\n===== 訓練模式執行完畢 =====")

    else:
        # ========================== 預測模式 ==========================
        print("===== 模式：載入模型 & 預測 =====")

        # --- 1. 載入預處理物件和元模型 ---
        print("載入預處理物件、元模型和特徵列表...")
        if not os.path.exists(MODEL_SAVE_DIR): print(f"錯誤：模型儲存目錄 '{MODEL_SAVE_DIR}' 不存在。請先運行訓練模式。"); exit()
        try:
            sc = joblib.load(SCALER_FILE)
            final_features = joblib.load(FINAL_FEATURES_FILE)
            meta_model = joblib.load(META_MODEL_FILE)
            if hasattr(meta_model, 'feature_names_in_'): model_names = list(meta_model.feature_names_in_)
            else: # 手動獲取模型名稱列表
                temp_classifier = Classifier(n_estimators=10, device=device, random_state=RANDOM_STATE_LIST[0])
                model_names = list(temp_classifier.models.keys()); del temp_classifier
            n_models = len(model_names)
            print(f"必要物件載入成功: Scaler, 特徵列表({len(final_features)}), 元模型, 基礎模型列表({n_models})")
        except FileNotFoundError as e: print(f"錯誤：找不到儲存檔案 ({e.filename})。請確保已成功運行訓練模式。"); exit()
        except Exception as e: print(f"錯誤：載入儲存檔案時出錯 - {e}"); exit()

        # --- 2. 載入並預處理新數據 ---
        print(f"載入並預處理新數據: {TEST_FILE_FOR_PREDICTION}")
        try:
            X_new = pd.read_csv(TEST_FILE_FOR_PREDICTION)
            new_test_ids = X_new['id'].copy() if 'id' in X_new.columns else None
            if 'id' in X_new.columns: X_new = X_new.drop(columns=['id'])

            # !!! ===> 關鍵：應用與訓練模式完全一致的預處理步驟 <=== !!!
            # 這裡需要非常小心地複製或重構訓練模式中的所有步驟
            # 以下是一個簡化示例，您需要填充完整邏輯
            print("應用預處理步驟 (請確保此處邏輯與訓練模式完全一致)...")

            # 步驟 1: create_extra_features
            X_new = create_extra_features(X_new.copy()) # 使用 .copy()

            # 步驟 2 & 3: 數值轉換 & 類別編碼
            # 這些步驟依賴於訓練數據來 fit 轉換器或計算映射
            # 最好的方法是在訓練模式儲存這些轉換器/映射，然後在這裡載入使用
            # 簡化處理：假設這些步驟對新數據影響不大或難以複製，先跳過
            # 但這會導致與訓練數據處理不一致！
            print("警告：預測模式下跳過了複雜的數值轉換和類別編碼步驟，可能導致結果不準確！建議儲存並載入轉換器/編碼器。")
            # 為了讓後續步驟能運行，需要確保訓練時產生的欄位存在，即使是填充預設值
            # 例如，如果訓練時產生了 'feature_count', 這裡也要有
            # for col in expected_cols_after_encoding: # 需要知道訓練後的欄位
            #      if col not in X_new.columns: X_new[col] = 0

            # 步驟 4: 算術特徵
            # 如果訓練時使用了 better_features，需要載入生成的特徵列表並計算
            # 簡化處理：跳過
            print("警告：預測模式下跳過了算術特徵生成。")

            # 步驟 5: 標準化 (使用載入的 sc)
            print("應用標準化...")
            features_to_scale_pred = [f for f in X_new.columns if f in sc.feature_names_in_]
            missing_scaler_features = list(set(sc.feature_names_in_) - set(X_new.columns))
            if missing_scaler_features:
                print(f"警告：新數據缺少以下用於標準化的欄位: {missing_scaler_features}。將用 0 填充。")
                for col in missing_scaler_features: X_new[col] = 0
            # 確保順序和欄位完全一致
            features_to_scale_pred_ordered = [f for f in sc.feature_names_in_ if f in X_new.columns] # 按 scaler 訓練時的順序
            X_new_scale_input = X_new[features_to_scale_pred_ordered]
            X_new_scale_input = X_new_scale_input.fillna(X_new_scale_input.mean()) # 填充 NaN
            X_new[features_to_scale_pred_ordered] = sc.transform(X_new_scale_input) # 應用標準化

            # 步驟 6: 特徵選擇 (使用載入的 final_features 列表)
            print("應用特徵選擇...")
            missing_final_features_pred = list(set(final_features) - set(X_new.columns))
            if missing_final_features_pred:
                 print(f"警告：處理後的新數據缺少最終特徵列表中的以下欄位: {missing_final_features_pred}。將用 0 填充。")
                 for col in missing_final_features_pred: X_new[col] = 0
            # 只保留最終選擇的特徵，並確保順序一致
            X_test_selected_new = X_new[final_features]

            print("新數據預處理和特徵選擇完成。維度:", X_test_selected_new.shape)
            # !!! ===> 預處理結束 <=== !!!

        except FileNotFoundError: print(f"錯誤：找不到新數據檔案 {TEST_FILE_FOR_PREDICTION}"); exit()
        except Exception as e: print(f"錯誤：處理新數據時出錯 - {e}"); import traceback; traceback.print_exc(); exit()


        # --- 3. 載入 Level 0 模型並生成 Level 1 特徵 ---
        print("\n載入 Level 0 模型並生成 Level 1 測試特徵...")
        test_predictions_level1 = np.zeros((len(X_test_selected_new), n_models))
        total_runs = len(RANDOM_STATE_LIST) * N_SPLITS

        for model_idx, name in enumerate(tqdm(model_names, desc="處理 Level 0 模型")):
            preds_for_this_model_type = []
            model_load_count = 0
            for seed in RANDOM_STATE_LIST:
                for fold in range(N_SPLITS):
                    model_filename = os.path.join(MODEL_SAVE_DIR, f'model_{name}_fold{fold+1}_seed{seed}.pkl')
                    try:
                        model = joblib.load(model_filename)
                        if hasattr(model, 'predict_proba'): pred = model.predict_proba(X_test_selected_new)[:, 1]
                        elif hasattr(model, 'predict'): pred = model.predict(X_test_selected_new).astype(float)
                        else: pred = np.full(len(X_test_selected_new), np.nan)
                        preds_for_this_model_type.append(pred)
                        model_load_count += 1
                    except FileNotFoundError: print(f"警告: 找不到模型檔案 {model_filename}，跳過。"); preds_for_this_model_type.append(np.full(len(X_test_selected_new), np.nan))
                    except Exception as e: print(f"警告: 載入或預測模型 {model_filename} 失敗 - {e}"); preds_for_this_model_type.append(np.full(len(X_test_selected_new), np.nan))

            # 計算平均預測
            if model_load_count > 0:
                 avg_pred = np.nanmean(np.array(preds_for_this_model_type), axis=0)
                 if np.isnan(avg_pred).all(): test_predictions_level1[:, model_idx] = 0.5
                 else: test_predictions_level1[:, model_idx] = np.nan_to_num(avg_pred, nan=0.5)
            else: test_predictions_level1[:, model_idx] = 0.5

        print("Level 1 測試特徵生成完成。")

        # --- 4. 使用載入的元模型進行最終預測 ---
        print("\n使用載入的元模型進行最終預測...")
        X_meta_test_new = pd.DataFrame(test_predictions_level1, columns=model_names)

        # (可選：如果訓練時加入了原始特徵，這裡也要加入)
        # ...

        try:
            final_predictions = meta_model.predict_proba(X_meta_test_new)[:, 1]
            print("最終預測完成。")
        except Exception as e: print(f"錯誤：使用元模型進行預測時出錯 - {e}"); final_predictions = np.full(len(X_test_selected_new), 0.5)

        # --- 5. 儲存預測結果 ---
        print(f"\n儲存預測結果到 {OUTPUT_FILE_PREDICTION}...")
        if new_test_ids is not None: output_df = pd.DataFrame({'id': new_test_ids, 'smoking': final_predictions})
        else: output_df = pd.DataFrame({'smoking': final_predictions})
        try:
            output_df.to_csv(OUTPUT_FILE_PREDICTION, index=False); print("預測結果儲存完畢。"); print(output_df.head())
        except Exception as e: print(f"錯誤：儲存預測結果時出錯 - {e}")

        print("\n===== 預測模式執行完畢 =====")