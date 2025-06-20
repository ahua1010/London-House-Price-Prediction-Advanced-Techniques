"""
倫敦房價預測 - 混合模型（趨勢分析 + 機器學習）
使用時間序列特徵結合機器學習進行房價預測
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def load_and_prepare_data():
    """載入並準備訓練和測試資料"""
    print("載入資料...")
    train_df = pd.read_csv('/kaggle/input/new-london-house-price/train.csv')
    test_df = pd.read_csv('/kaggle/input/new-london-house-price/test.csv')
    
    # 為測試集添加空的價格欄位
    test_df['price'] = np.nan
    
    return train_df, test_df


def create_time_features(data_list):
    """創建時間相關特徵"""
    print("創建時間特徵...")
    for data in data_list:
        # 創建時間索引
        data['time'] = pd.to_datetime(dict(
            year=data['sale_year'], 
            month=data['sale_month'], 
            day=15
        ))
        data['time'] = data['time'].dt.to_period('M')
        
        # 創建數值型時間特徵
        data['time_numeric'] = (
            (data['time'].dt.to_timestamp() - data['time'].min().to_timestamp()) / 
            np.timedelta64(1, 'D')
        )
    
    return data_list


def preprocess_address_features(data_list):
    """處理地址相關特徵"""
    print("處理地址特徵...")
    for data in data_list:
        # 提取街道資訊
        data['street'] = data['fullAddress'].apply(
            lambda address: ' '.join(address.split(',')[-3].split(' ')[-2:])
        )
        
        # 處理郵遞區號
        data['postcode'] = data['postcode'].apply(
            lambda postcode: postcode.split(' ')[1]
        )
        
        # 移除國家欄位（所有資料都是同一個國家）
        data.drop('country', axis=1, inplace=True)
    
    return data_list


def impute_missing_values_with_strategy(data_list, column_name, strategy='most_frequent'):
    """使用指定策略填補缺失值"""
    print(f"填補 {column_name} 的缺失值（策略：{strategy}）...")
    
    # 從訓練資料學習填補策略
    train_data = data_list[0]  # 第一個是訓練資料
    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(train_data[[column_name]])
    
    # 對所有資料集應用填補
    for data in data_list:
        data[column_name] = imputer.transform(data[[column_name]]).ravel()
    
    return data_list


def impute_with_regression(data_list, target_column, feature_column):
    """使用回歸模型填補缺失值"""
    print(f"使用 {feature_column} 預測填補 {target_column} 的缺失值...")
    
    train_data = data_list[0]
    test_data = data_list[1]
    
    # 準備完整的訓練資料
    complete_train_data = train_data.dropna(subset=[target_column, feature_column])
    X_train = complete_train_data[[feature_column]]
    y_train = complete_train_data[target_column]
    
    # 訓練回歸模型
    regression_model = Ridge()
    regression_model.fit(X_train, y_train)
    
    # 填補訓練集的缺失值
    missing_train_mask = train_data[target_column].isna()
    if missing_train_mask.any():
        missing_train_features = train_data.loc[missing_train_mask, [feature_column]]
        train_data.loc[missing_train_mask, target_column] = regression_model.predict(missing_train_features)
    
    # 填補測試集的缺失值
    missing_test_mask = test_data[target_column].isna()
    if missing_test_mask.any():
        missing_test_features = test_data.loc[missing_test_mask, [feature_column]]
        test_data.loc[missing_test_mask, target_column] = regression_model.predict(missing_test_features)
    
    return data_list


def handle_missing_values(data_list):
    """處理所有缺失值"""
    print("開始處理缺失值...")
    
    # 使用最頻繁值填補面積
    data_list = impute_missing_values_with_strategy(data_list, 'floorAreaSqM')
    
    # 使用面積預測浴室數量
    data_list = impute_with_regression(data_list, 'bathrooms', 'floorAreaSqM')
    
    # 使用面積預測臥室數量
    data_list = impute_with_regression(data_list, 'bedrooms', 'floorAreaSqM')
    
    # 使用最頻繁值填補其他類別特徵
    categorical_columns = ['livingRooms', 'tenure', 'propertyType', 'currentEnergyRating']
    for column in categorical_columns:
        data_list = impute_missing_values_with_strategy(data_list, column)
    
    return data_list


def create_time_series_features(train_data, test_data):
    """創建時間序列特徵"""
    print("創建時間序列特徵...")
    
    # 結合訓練集和測試集的時間索引，以涵蓋所有時間點
    combined_index = train_data.index.union(test_data.index).unique()

    # 創建確定性過程（趨勢、季節性、週期性）
    deterministic_process = DeterministicProcess(
        index=combined_index,
        constant=True,        # 常數項
        seasonal=True,        # 季節性
        order=12,            # 趨勢階數
        drop=True,           # 移除共線性
        additional_terms=[CalendarFourier(freq="QE", order=4)],  # 季度傅立葉項
    )
    
    # 為整個時間範圍生成特徵
    time_features = deterministic_process.in_sample()
    
    # 分別為訓練資料和測試資料添加時間序列特徵
    train_data = train_data.join(time_features, how='left')
    test_data = test_data.join(time_features, how='left')
    
    # 確保測試資料的索引名稱正確
    test_data.index.name = 'time'
    
    # 移除預測相關的打印信息，因為這不再是預測任務
    print("時間序列特徵已為訓練集和測試集創建。")

    return train_data, test_data, time_features.columns.tolist()


def create_additional_features(data_list):
    """創建額外的特徵"""
    print("創建額外特徵...")
    
    for data in data_list:
        # 總房間數 = 臥室 + 起居室
        data['rooms'] = data['bedrooms'] + data['livingRooms']
    
    return data_list


class CustomEncoder(BaseEstimator, TransformerMixin):
    """
    自定義編碼器，用於處理類別特徵
    包括：街道、郵遞區號、outcode、tenure、緯度/經度分箱、房產類型、能源評級
    """
    
    def __init__(self):
        self.target_mean_encoders = {}  # 目標編碼器
        self.fallback_values = {}       # 備用值
        self.bin_encoders = {}          # 分箱編碼器
        self.ordinal_encoders = {}      # 順序編碼器

    def fit(self, X, y=None):
        """學習編碼規則"""
        X_copy = X.copy()
        X_copy['price'] = y
        
        # 目標編碼特徵
        target_encoding_features = ['street', 'postcode', 'outcode', 'tenure', 'propertyType']
        for feature in target_encoding_features:
            self.target_mean_encoders[feature] = X_copy.groupby(feature)['price'].mean()
            self.fallback_values[feature] = self.target_mean_encoders[feature].mean()
        
        # 緯度分箱編碼
        latitude_bins = pd.cut(X_copy['latitude'], bins=10, retbins=True)[1]
        self.bin_encoders['latitudeBins'] = latitude_bins
        latitude_labels = pd.cut(X_copy['latitude'], bins=latitude_bins, include_lowest=True)
        self.bin_encoders['latitude_label_encoder'] = LabelEncoder().fit(latitude_labels)
        
        # 經度分箱編碼
        longitude_bins = pd.cut(X_copy['longitude'], bins=10, retbins=True)[1]
        self.bin_encoders['longitudeBins'] = longitude_bins
        longitude_labels = pd.cut(X_copy['longitude'], bins=longitude_bins, include_lowest=True)
        self.bin_encoders['longitude_label_encoder'] = LabelEncoder().fit(longitude_labels)
        
        # 能源評級順序編碼
        energy_rating_order = [['G', 'F', 'E', 'D', 'C', 'B', 'A']]
        self.ordinal_encoders['currentEnergyRating'] = OrdinalEncoder(
            categories=energy_rating_order
        ).fit(X_copy[['currentEnergyRating']])
        
        return self

    def transform(self, X):
        """應用編碼轉換"""
        X_transformed = X.copy()
        
        # 目標編碼
        target_encoding_features = ['street', 'postcode', 'outcode', 'tenure', 'propertyType']
        for feature in target_encoding_features:
            X_transformed[feature] = X_transformed[feature].map(self.target_mean_encoders[feature])
            X_transformed[feature] = X_transformed[feature].fillna(self.fallback_values[feature])
        
        # 緯度分箱
        latitude_bins = pd.cut(
            X_transformed['latitude'], 
            bins=self.bin_encoders['latitudeBins'], 
            include_lowest=True, 
            right=True
        )
        X_transformed['latitudeBins'] = self.bin_encoders['latitude_label_encoder'].transform(latitude_bins)
        
        # 經度分箱
        longitude_bins = pd.cut(
            X_transformed['longitude'], 
            bins=self.bin_encoders['longitudeBins'], 
            include_lowest=True, 
            right=True
        )
        X_transformed['longitudeBins'] = self.bin_encoders['longitude_label_encoder'].transform(longitude_bins)
        
        # 能源評級順序編碼
        X_transformed['currentEnergyRating'] = self.ordinal_encoders['currentEnergyRating'].transform(
            X_transformed[['currentEnergyRating']]
        )
        
        return X_transformed


class HybridModel(BaseEstimator, RegressorMixin):
    """
    混合模型：結合趨勢模型和機器學習模型
    - 趨勢模型：處理時間序列特徵
    - 機器學習模型：處理殘差和其他特徵
    """
    
    def __init__(self, trend_model, machine_model, trend_cols, machine_cols, all_columns):
        self.trend_model = trend_model
        self.machine_model = machine_model
        self.trend_cols = trend_cols
        self.machine_cols = machine_cols
        self.all_columns = all_columns

    def fit(self, X, y):
        """訓練混合模型"""
        # 對目標變量進行對數轉換以穩定方差
        y_log = np.log1p(y)
        
        # 確保輸入是 DataFrame 格式
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.all_columns)
        
        # 分離趨勢特徵和機器學習特徵
        trend_features = X[self.trend_cols]
        machine_features = X[self.machine_cols]
        
        # 1. 訓練趨勢模型
        self.trend_model.fit(trend_features, y_log)
        
        # 2. 計算趨勢預測的殘差
        trend_predictions = self.trend_model.predict(trend_features)
        residual = y_log - trend_predictions
        
        # 3. 用機器學習模型學習殘差
        self.machine_model.fit(machine_features, residual)
        
        return self

    def predict(self, X):
        """進行預測"""
        # 確保輸入是 DataFrame 格式
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.all_columns)
        
        # 分離特徵
        trend_features = X[self.trend_cols]
        machine_features = X[self.machine_cols]
        
        # 獲取趨勢預測和機器學習預測
        trend_predictions = self.trend_model.predict(trend_features)
        machine_predictions = self.machine_model.predict(machine_features)
        
        # 組合預測結果並反向對數轉換
        combined_predictions = trend_predictions + machine_predictions
        return np.expm1(combined_predictions)


def prepare_features(train_data, test_data, time_series_features):
    """準備特徵集合"""
    print("準備特徵集合...")
    
    # 時間序列特徵（用於趨勢模型）
    trend_features = time_series_features
    
    # 機器學習特徵（用於殘差模型）
    machine_learning_features = [
        'street', 'postcode', 'outcode', 'latitudeBins', 'longitudeBins',
        'bathrooms', 'bedrooms', 'rooms', 'floorAreaSqM', 'livingRooms',
        'tenure', 'propertyType', 'currentEnergyRating'
    ]
    
    # 準備訓練特徵和目標
    X_train = train_data.drop('price', axis=1)
    y_train = train_data['price']
    
    # 標準化時間序列特徵
    scaler = StandardScaler()
    X_train[trend_features] = scaler.fit_transform(X_train[trend_features])
    test_data[trend_features] = scaler.transform(test_data[trend_features])
    
    return X_train, y_train, trend_features, machine_learning_features


def create_and_tune_model(X_train, y_train, trend_features, machine_learning_features):
    """創建並調優模型"""
    print("創建混合模型並進行超參數調優...")
    
    # 定義模型管道
    model_pipeline = {
        'HybridModel': Pipeline([
            ('Encoder', CustomEncoder()),
            ('Model', HybridModel(
                trend_model=Ridge(),
                machine_model=XGBRegressor(),
                trend_cols=trend_features,
                machine_cols=machine_learning_features,
                all_columns=X_train.columns
            ))
        ]),
    }
    
    # 定義超參數搜索空間
    hyperparameter_grid = {
        'HybridModel': {
            'Model__trend_model__alpha': [0.01, 0.1],
            'Model__machine_model__n_estimators': [500],
            'Model__machine_model__max_depth': [9],
            'Model__machine_model__learning_rate': [0.01, 0.005, 0.1],
        }
    }
    
    # 定義交叉驗證策略：隨機 K 折
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # 進行網格搜索
    best_models = {}
    for model_name, pipeline in model_pipeline.items():
        print(f"調優 {model_name}...")
        
        # 使用 KFold 進行隨機分割交叉驗證
        grid_search = GridSearchCV(
            pipeline, 
            hyperparameter_grid[model_name], 
            cv=kfold, 
            scoring='neg_mean_absolute_error', 
            n_jobs=-1, 
            verbose=2, 
            error_score='raise'
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"{model_name} 最佳參數: {grid_search.best_params_}")
        print(f"{model_name} 最佳 MAE: {-grid_search.best_score_:.4f}")
        
        best_models[model_name] = grid_search.best_estimator_
    
    return best_models


def create_ensemble_model(best_models, X_train, y_train):
    """創建集成模型"""
    print("創建集成模型...")
    
    # 準備集成模型的估計器列表
    ensemble_estimators = [
        ('HybridModel', best_models['HybridModel']),
    ]
    
    # 創建投票回歸器
    ensemble_model = VotingRegressor(estimators=ensemble_estimators)
    ensemble_model.fit(X_train, y_train)
    
    print(f"集成模型: {ensemble_model}")
    
    return ensemble_model


def evaluate_model(model, X_train, y_train):
    """評估模型性能"""
    print("評估模型性能...")
    
    # 預測訓練集
    train_predictions = model.predict(X_train)
    
    # 計算評估指標
    mae = mean_absolute_error(y_train, train_predictions)
    rmse = mean_squared_error(y_train, train_predictions, squared=False)
    r2 = r2_score(y_train, train_predictions)
    
    print(f"[訓練集] MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    return mae, rmse, r2


def generate_submission(model, test_data):
    """生成提交檔案"""
    print("生成提交檔案...")
    
    # 載入提交模板
    submission = pd.read_csv('/kaggle/input/new-london-house-price/sample_submission.csv')
    
    # 進行預測
    test_features = test_data.drop('price', axis=1)
    submission['price'] = model.predict(test_features)
    
    # 儲存提交檔案
    submission.to_csv('submission.csv', index=False)
    print("提交檔案已儲存為 submission.csv")


def main():
    """主要執行流程"""
    print("=== 倫敦房價預測 - 混合模型 ===")
    
    # 1. 載入資料
    train_df, test_df = load_and_prepare_data()
    data_list = [train_df, test_df]
    
    # 2. 創建時間特徵
    data_list = create_time_features(data_list)
    
    # 3. 設定時間索引
    train_df = data_list[0].set_index('time')
    test_df = data_list[1].set_index('time')
    data_list = [train_df, test_df]
    
    # 4. 預處理地址特徵
    data_list = preprocess_address_features(data_list)
    
    # 5. 處理缺失值
    data_list = handle_missing_values(data_list)
    train_df, test_df = data_list[0], data_list[1]
    
    # 6. 創建時間序列特徵
    train_df, test_df, time_series_features = create_time_series_features(train_df, test_df)
    
    # 7. 創建額外特徵
    data_list = create_additional_features([train_df, test_df])
    train_df, test_df = data_list[0], data_list[1]
    
    # 8. 準備特徵
    X_train, y_train, trend_features, machine_learning_features = prepare_features(
        train_df, test_df, time_series_features
    )
    
    # 9. 創建並調優模型
    best_models = create_and_tune_model(X_train, y_train, trend_features, machine_learning_features)
    
    # 10. 創建集成模型
    final_model = create_ensemble_model(best_models, X_train, y_train)
    
    # 11. 評估模型
    evaluate_model(final_model, X_train, y_train)
    
    # 12. 生成提交檔案
    generate_submission(final_model, test_df)
    
    print("=== 程序執行完成 ===")


# 執行主程序
if __name__ == "__main__":
    main()