import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.seasonal import STL
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesAwareMetaModel(BaseEstimator):
    """方法1：添加時間序列特徵的元模型"""
    def __init__(self, base_meta_model, time_features: List[str], time_window: int = 3):
        self.base_meta_model = base_meta_model
        self.time_features = time_features
        self.time_window = time_window
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TimeSeriesAwareMetaModel':
        # 創建時間窗口特徵
        time_features = self._create_time_window_features(X)
        # 合併原始特徵和時間窗口特徵
        X_with_time = pd.concat([X, time_features], axis=1)
        # 訓練基礎元模型
        self.base_meta_model.fit(X_with_time, y)
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        time_features = self._create_time_window_features(X)
        X_with_time = pd.concat([X, time_features], axis=1)
        return self.base_meta_model.predict(X_with_time)
    
    def _create_time_window_features(self, X: pd.DataFrame) -> pd.DataFrame:
        time_features = pd.DataFrame(index=X.index)
        for feature in self.time_features:
            for i in range(1, self.time_window + 1):
                time_features[f'{feature}_lag_{i}'] = X[feature].shift(i)
        return time_features.fillna(method='bfill')

class HierarchicalTimeSeriesMetaModel(BaseEstimator):
    """方法2：分層時間序列預測"""
    def __init__(self, base_meta_model, period: int = 12):
        self.base_meta_model = base_meta_model
        self.period = period
        self.trend_model = None
        self.seasonal_model = None
        self.residual_model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'HierarchicalTimeSeriesMetaModel':
        # 確保數據按時間順序排列
        if 'sale_year' in X.columns and 'sale_month' in X.columns:
            sort_cols = ['sale_year', 'sale_month']
            if 'sale_day' in X.columns:
                sort_cols.append('sale_day')
            X = X.sort_values(sort_cols)
            y = y[X.index]

        print("[HierarchicalMeta] X shape:", X.shape)
        # print("[HierarchicalMeta] y shape:", y.shape)
        print("[HierarchicalMeta] X head:\n", X.head())
        # print("[HierarchicalMeta] y head:\n", y.head())
        # print("[HierarchicalMeta] X 全為NaN:", X.isnull().all().all())
        # print("[HierarchicalMeta] y 全為NaN:", y.isnull().all())
        # print("[HierarchicalMeta] X dtypes:\n", X.dtypes)

        try:
            # 時間序列分解
            stl = STL(y, period=self.period)
            result = stl.fit()
            trend, seasonal, residual = result.trend, result.seasonal, result.resid
            print("[HierarchicalMeta] STL trend head:\n", trend.head())
            print("[HierarchicalMeta] STL seasonal head:\n", seasonal.head())
            print("[HierarchicalMeta] STL residual head:\n", residual.head())
            # print("[HierarchicalMeta] trend 全為NaN:", trend.isnull().all())
            # print("[HierarchicalMeta] seasonal 全為NaN:", seasonal.isnull().all())
            # print("[HierarchicalMeta] residual 全為NaN:", residual.isnull().all())
            # print("[HierarchicalMeta] trend shape:", trend.shape)

            # 訓練各個組件的模型
            self.trend_model = clone(self.base_meta_model)
            self.seasonal_model = clone(self.base_meta_model)
            self.residual_model = clone(self.base_meta_model)

            print("[HierarchicalMeta] trend_model.fit ...")
            self.trend_model.fit(X, trend)
            print("[HierarchicalMeta] seasonal_model.fit ...")
            self.seasonal_model.fit(X, seasonal)
            print("[HierarchicalMeta] residual_model.fit ...")
            self.residual_model.fit(X, residual)
            print("[HierarchicalMeta] fit 完成")
            return self
        except Exception as e:
            print(f"[HierarchicalMeta] 分層時間序列模型訓練時發生錯誤: {e}")
            print("[HierarchicalMeta] X info:")
            print(X.info())
            print("[HierarchicalMeta] y describe:\n", y.describe())
            print("[HierarchicalMeta] X describe:\n", X.describe())
            raise e
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            trend_pred = self.trend_model.predict(X)
            seasonal_pred = self.seasonal_model.predict(X)
            residual_pred = self.residual_model.predict(X)
            
            return trend_pred + seasonal_pred + residual_pred
        except Exception as e:
            print(f"分層時間序列模型預測時發生錯誤: {e}")
            # 如果預測失敗，只返回趨勢預測
            return self.trend_model.predict(X)

class TimeSeriesCrossValidationMetaModel(BaseEstimator):
    """方法3：時間序列交叉驗證的元模型"""
    def __init__(self, base_meta_model, n_splits: int = 5):
        self.base_meta_model = base_meta_model
        self.n_splits = n_splits
        self.meta_models = []
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TimeSeriesCrossValidationMetaModel':
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            meta_model = clone(self.base_meta_model)
            meta_model.fit(X_train, y_train)
            self.meta_models.append(meta_model)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = []
        weights = []
        
        for i, model in enumerate(self.meta_models):
            pred = model.predict(X)
            predictions.append(pred)
            # 根據時間遠近給予不同權重
            weight = 1 / (i + 1)
            weights.append(weight)
        
        # 加權平均預測
        final_pred = np.average(predictions, weights=weights, axis=0)
        return final_pred

class EnsembleTimeSeriesMetaModel(BaseEstimator):
    """方法4：集成時間序列元模型（結合前三種方法）"""
    def __init__(self, base_meta_model, time_features: List[str], n_splits: int = 5, period: int = 12):
        self.base_meta_model = base_meta_model
        self.time_features = time_features
        self.n_splits = n_splits
        self.period = period
        
        # 初始化三種元模型
        self.ts_aware_model = TimeSeriesAwareMetaModel(
            clone(base_meta_model), 
            time_features=time_features
        )
        self.hierarchical_model = HierarchicalTimeSeriesMetaModel(
            clone(base_meta_model),
            period=period
        )
        self.ts_cv_model = TimeSeriesCrossValidationMetaModel(
            clone(base_meta_model),
            n_splits=n_splits
        )
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnsembleTimeSeriesMetaModel':
        # 訓練三種元模型
        self.ts_aware_model.fit(X, y)
        self.hierarchical_model.fit(X, y)
        self.ts_cv_model.fit(X, y)
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # 獲取三種模型的預測
        pred1 = self.ts_aware_model.predict(X)
        pred2 = self.hierarchical_model.predict(X)
        pred3 = self.ts_cv_model.predict(X)
        
        # 簡單平均
        return (pred1 + pred2 + pred3) / 3 