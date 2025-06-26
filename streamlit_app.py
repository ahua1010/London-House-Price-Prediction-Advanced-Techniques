import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
import time

warnings.filterwarnings('ignore')

# --- 核心函式與類別 (從 visualize_map.py 遷移) ---

def create_time_features(df):
    df['time'] = pd.to_datetime(df['sale_year'].astype(str) + '-' + df['sale_month'].astype(str) + '-01')
    df['time_numeric'] = (df['time'] - df['time'].min()).dt.days
    return df

class HybridModel(BaseEstimator, RegressorMixin):
    def __init__(self, trend_model, ml_model):
        self.trend_model = trend_model
        self.ml_model = ml_model
        self.trend_cols = None

    def fit(self, X, y):
        self.trend_cols = [col for col in X.columns if col.startswith('trend_') or col == 'const']
        if not self.trend_cols: self.trend_cols = ['time_numeric']
        self.trend_model.fit(X[self.trend_cols], y)
        y_trend = self.trend_model.predict(X[self.trend_cols])
        y_residuals = y - y_trend
        self.ml_model.fit(X, y_residuals)
        return self

    def predict(self, X):
        y_trend = self.trend_model.predict(X[self.trend_cols])
        y_residuals = self.ml_model.predict(X)
        return y_trend + y_residuals

def get_gradient_color(error_pct):
    max_error_for_color = 40.0
    scaled_error = min(error_pct, max_error_for_color) / max_error_for_color
    if scaled_error <= 0.5:
        t = scaled_error * 2
        r, g, b = int(t * 255), 255, 0
    else:
        t = (scaled_error - 0.5) * 2
        r, g, b = 255, int(255 * (1 - t)), 0
    return [r, g, b, 160]

# --- Streamlit 應用主體 ---

st.set_page_config(layout="wide", page_title="倫敦房價視覺化")

st.title('倫敦房價互動式 3D 地圖')

# 重要的修正：從舊檔案或直接寫入您的 API KEY
MAPBOX_API_KEY = "Your API KEY"

@st.cache_data
def load_and_process_data():
    df = pd.read_csv('train.csv')
    df = df.sample(n=60000, random_state=42)
    
    # 進行特徵工程
    df = create_time_features(df)
    df['propertyType_original'] = df['propertyType'] # 保留原始房產類型用於篩選

    categorical_features = ['propertyType', 'outcode']
    numerical_features = ['bedrooms', 'bathrooms', 'latitude', 'longitude', 'time_numeric']
    
    for col in categorical_features + numerical_features:
        if df[col].isnull().any():
            if df[col].dtype.kind in 'fi':
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        
    # 訓練模型並得到預測
    features = categorical_features + numerical_features
    X = df[features]
    y = df['price']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    trend_model = LinearRegression()
    ml_model = LGBMRegressor(random_state=42)
    
    hybrid_model = HybridModel(trend_model=trend_model, ml_model=ml_model)
    with st.spinner('正在訓練模型並生成預測...'):
        hybrid_model.fit(X_train, y_train)
        predictions = hybrid_model.predict(X_val)

    results_df = X_val.copy()
    results_df['price'] = y_val
    results_df['prediction'] = predictions
    results_df['propertyType'] = df.loc[X_val.index, 'propertyType_original'] # 換回原始名稱
    
    results_df['error_pct'] = 100 * np.abs(results_df['price'] - results_df['prediction']) / results_df['price']
    results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    results_df.dropna(subset=['error_pct', 'latitude', 'longitude'], inplace=True)

    results_df['elevation_val'] = np.sqrt(results_df['price'])
    results_df['color'] = results_df['error_pct'].apply(get_gradient_color)

    return results_df

results = load_and_process_data()

# --- 側邊欄 UI 控制項 ---

st.sidebar.header('地圖控制項')

map_style = st.sidebar.selectbox(
    '選擇地圖風格',
    ('mapbox://styles/mapbox/dark-v9', 'mapbox://styles/mapbox/light-v9', 'mapbox://styles/mapbox/streets-v11', 'mapbox://styles/mapbox/satellite-v9'),
    format_func=lambda x: x.split('/')[-1].replace('-v9', '').replace('-v11', '').capitalize() # 顯示簡化的名稱
)

property_types = st.sidebar.multiselect(
    '篩選房產類型',
    options=results['propertyType'].unique(),
    default=results['propertyType'].unique()
)

# --- 根據篩選過濾資料 ---

filtered_data = results[results['propertyType'].isin(property_types)]

# Sample data if it's too large
if len(filtered_data) > 3000:
    display_data = filtered_data.sample(n=3000, random_state=42)
else:
    display_data = filtered_data

st.info(f"顯示 {len(display_data)} / {len(results)} 個房產")

# --- 地圖渲染 ---

st.pydeck_chart(pdk.Deck(
    map_style=map_style,
    api_keys={'mapbox': MAPBOX_API_KEY},
    initial_view_state=pdk.ViewState(
        latitude=display_data['latitude'].mean(),
        longitude=display_data['longitude'].mean(),
        zoom=10,
        pitch=50,
        min_zoom=8,
        max_zoom=5
    ),
    layers=[
        pdk.Layer(
            'ColumnLayer',
            data=display_data,
            get_position='[longitude, latitude]',
            get_elevation='elevation_val',
            elevation_scale=2,
            radius=40,
            get_fill_color='color',
            pickable=True,
            auto_highlight=True,
        ),
    ],
    tooltip={
        "html": "<b>房價:</b> £{price}<br/>"
                "<b>預測價:</b> £{prediction}<br/>"
                "<b>誤差:</b> {error_pct}%",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }
))

# --- 原生圖例 ---
st.sidebar.info("提示：按住 Ctrl 鍵並拖曳滑鼠來旋轉與傾斜地圖。")
st.sidebar.header("圖例")
legend_html = """
<div style="background: rgba(20, 20, 20, 0.7); color: white; padding: 10px; font-size: 12px; border-radius: 5px; border: 1px solid #555;">
    <h4 style="margin-top: 0; margin-bottom: 10px; text-align: center;">Prediction Accuracy</h4>
    <div style="width: 100%; height: 15px; background: linear-gradient(to right, rgb(0,255,0), rgb(255,255,0), rgb(255,0,0)); margin-bottom: 5px; border: 1px solid #999;"></div>
    <div style="display: flex; justify-content: space-between;">
        <span style="font-size: 10px;">Low Error</span>
        <span style="font-size: 10px;">High Error</span>
    </div>
</div>
"""
st.sidebar.markdown(legend_html, unsafe_allow_html=True) 
