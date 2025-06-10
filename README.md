# 房價預測系統

## 專案概觀
本專案旨在開發一個基於機器學習的房價預測系統。透過分析歷史房產數據，建立一個包含特徵工程、模型訓練與預測的完整管線，核心目標是達到高準確度的預測能力。

## 技術棧
- **語言:** Python 3.8+
- **核心函式庫:**
  - `pandas`, `numpy`: 資料處理
  - `scikit-learn`: 機器學習工具
  - `xgboost`, `lightgbm`, `catboost`: 梯度提升模型
  - `joblib`, `pickle`: 模型序列化
  - `matplotlib`, `seaborn`: 資料視覺化

## 功能開發狀態追蹤表
| 功能或模組名稱 | 狀態 | 用途 | 核心功能 | 技術實現 | 用戶流程(導航) | 建議文件路徑 |
|---|---|---|---|---|---|---|
| 資料載入模組 | ✅ 已完成 | 從 CSV 檔案載入訓練與測試資料集。 | 讀取 `train.csv` 和 `test.csv`。 | 使用 `pandas.read_csv`。 | N/A (後端模組) | `src/data_loader.py` |
| 特徵工程模組 | ✅ 已完成 | 清理數據、處理缺失值、並從現有數據中創造新特徵以提升模型效能。 | 缺失值插補、時間/地理特徵生成、類別特徵編碼、自動化算術特徵組合。 | `impute_feature_with_lgbm`, `create_derived_features`, `process_categorical_features` | N/A (後端模組) | `src/feature_engineering.py` |
| 模型訓練與評估模組 | ✅ 已完成 | 訓練、評估並儲存機器學習模型。 | 使用時間序列交叉驗證進行模型堆疊 (Stacking)，包含特徵選擇、基礎模型訓練、元模型訓練及模型持久化。 | `run_time_series_stacking` 函式，使用 `TimeSeriesSplit`, `LightGBM` (特徵選擇), `XGBoost`, `CatBoost`, `ExtraTreesRegressor` (元模型)。 | N/A (後端模組) | `src/model_pipeline.py` |
| 預測管線模組 | ✅ 已完成 | 載入已訓練的模型，對新的測試資料進行預測，並產生提交檔案。 | 載入最新模型、應用特徵工程、執行堆疊預測、生成 submission.csv。 | `load_latest_model` 函式載入 `config.pkl`，`predict` 函式協調整個預測流程。 | N/A (後端模組) | `src/predict.py` |

## 技術實現細節

### 特徵工程模組 詳細設計
此模組是提升模型預測能力的关键。它不只進行標準的數據清理，還採用了基於模型的先進技術來處理缺失值。
- **LGBM 缺失值插補:** 對於 `floorAreaSqM`, `bedrooms`, `bathrooms` 等重要特徵的缺失值，並非使用簡單的中位數或均值填充，而是獨立訓練 LightGBM 模型來預測這些缺失值。此方法利用了特徵間的相關性，能提供更精確的插補結果。
- **自動化特徵生成:** 腳本會自動探索特徵間的算術組合（加、減、乘、除），並透過交叉驗證評估其有效性，只保留對模型有益的組合特徵。
- **衍生特徵:** 創造了多種衍生特徵，包括時間相關（如月份的正弦/餘弦轉換）、房產結構（如總房間數）和地理位置特徵。

### 模型訓練與評估模組 詳細設計
此模組採用了時間序列交叉驗證和模型堆疊（Stacking）兩種高級策略，以確保模型的穩健性和準確性。
- **時間序列交叉驗證 (`TimeSeriesSplit`):** 傳統的隨機交叉驗證不適用於具時間性的房價數據。此專案採用時間序列切分，確保驗證集的時間永遠在訓練集之後，模擬真實世界的預測場景，防止數據洩漏。
- **模型堆疊 (Stacking):**
  1.  **基礎模型層:** 在交叉驗證的每一摺中，會訓練多個不同的基礎模型（如 LightGBM, XGBoost, CatBoost）。
  2.  **元特徵生成:** 將每個基礎模型對驗證集的預測結果（稱為 Out-of-Fold 預測）儲存起來，作為訓練元模型的「新特徵」。
  3.  **元模型層:** 最後，使用這些元特徵來訓練一個元模型（`ExtraTreesRegressor`），由它來做出最終的預測。這種方法能有效結合多個模型的優點，提升整體預測的準確度。
- **模型持久化:** 訓練完成後，所有必要的物件（包括特徵工程設定、特徵列表、所有基礎模型、元模型）都會被打包序列化到一個 `config.pkl` 檔案中，方便預測管線直接調用。

## 測試案例摘要
目前專案未包含自動化測試案例。建議未來導入 `pytest` 以確保各模組功能的穩定性，特別是針對 `feature_engineering.py` 中的複雜轉換邏輯和 `model_pipeline.py` 的訓練流程。

## 系統架構
```
src/
├── data_processing/      # 數據處理模塊
├── feature_engineering/  # 特徵工程模塊
├── model_training/       # 模型訓練模塊
├── evaluation/          # 模型評估模塊
└── utils/              # 工具函數模塊
```

## 特徵工程模塊詳解

### 1. 房產結構特徵
- **面積特徵**
  - `floorAreaSqM`: 使用 LGBM 模型進行缺失值插補
  - `floorAreaSqM_log`: 對面積進行對數轉換
  - `floorAreaSqM_is_missing`: 面積缺失指示符

- **房間特徵**
  - `bedrooms`: 臥室數量，使用 LGBM 模型插補
  - `bathrooms`: 浴室數量，使用 LGBM 模型插補
  - `livingRooms`: 客廳數量，使用 LGBM 模型插補
  - `total_rooms`: 總房間數（臥室+浴室+客廳）
  - 各房間特徵的缺失指示符

### 2. 房產類型特徵
- `propertyType`: 房產類型
  - 使用眾數填充缺失值
  - 進行編碼轉換
- `tenure`: 產權類型
  - 使用眾數填充缺失值
  - 進行編碼轉換

### 3. 時間特徵
- `month`: 月份
  - `sin_month`: 月份的正弦轉換
  - `cos_month`: 月份的餘弦轉換
- `months_since_start`: 距離起始時間的月數
- 時間與房產類型的交互特徵
- 時間與地區的交互特徵

### 4. 位置特徵
- `outcode`: 郵政編碼前綴
  - 進行編碼轉換
- `latitude`: 緯度
- `longitude`: 經度

### 5. 能源評級特徵
- `currentEnergyRating`: 當前能源評級
  - 進行編碼轉換
  - 使用全局均值填充缺失值

### 6. 特徵選擇
- 使用 LightGBM 進行特徵選擇，最終選出 30 個特徵用於訓練和預測。

## 使用說明

### 環境要求
- Python 3.8+
- 相關依賴包（見 requirements.txt）

### 安裝步驟
1. 克隆專案
2. 安裝依賴：`pip install -r requirements.txt`
3. 配置環境變數

### 運行方式
1. 數據處理：`python src/data_processing/process_data.py`
2. 特徵工程：`python src/feature_engineering/engineer_features.py`
3. 模型訓練：`python src/model_training/train_model.py`
4. 模型評估：`python src/evaluation/evaluate_model.py`

## 注意事項
1. 確保數據文件放在正確的目錄
2. 檢查配置文件中的參數設置
3. 注意模型訓練的資源需求

## 未來計劃
1. 添加更多特徵工程方法
2. 優化模型性能
3. 增加模型解釋性分析
4. 開發 Web 界面

## 貢獻指南
歡迎提交 Pull Request 或提出 Issue。

## 授權說明
MIT License 