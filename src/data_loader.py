import pandas as pd
import os


def load_data(
    train_file_name="train.csv",
    test_file_name="test.csv",
    base_path=".",
):
    """
    載入位於指定路徑的訓練和測試 CSV 檔案。

    參數:
    train_file_name (str): 訓練數據 CSV 檔案的名稱。
    test_file_name (str): 測試數據 CSV 檔案的名稱。
    base_path (str): CSV 檔案所在的基礎路徑。

    返回:
    tuple: (pandas.DataFrame, pandas.DataFrame)，分別為訓練數據和測試數據。
    如果檔案不存在，則對應的 DataFrame 為 None。
    """
    train_path = os.path.join(base_path, train_file_name)
    test_path = os.path.join(base_path, test_file_name)

    train_df = None
    test_df = None

    try:
        train_df = pd.read_csv(train_path)
        print(f"成功載入訓練數據: {train_path}")
    except FileNotFoundError:
        print(f"錯誤: 訓練數據檔案未找到於 {train_path}")
    except Exception as e:
        print(f"載入訓練數據時發生錯誤 ({train_path}): {e}")

    try:
        test_df = pd.read_csv(test_path)
        print(f"成功載入測試數據: {test_path}")
    except FileNotFoundError:
        print(f"錯誤: 測試數據檔案未找到於 {test_path}")
    except Exception as e:
        print(f"載入測試數據時發生錯誤 ({test_path}): {e}")

    return train_df, test_df


if __name__ == "__main__":
    # 範例使用方式 (當直接執行此腳本時)
    # 假設 train.csv 和 test.csv 在專案根目錄
    train_data, test_data = load_data()

    if train_data is not None:
        print("\n訓練數據概覽:")
        print(train_data.head())
        print("\n訓練數據資訊:")
        train_data.info()

    if test_data is not None:
        print("\n測試數據概覽:")
        print(test_data.head())
        print("\n測試數據資訊:")
        test_data.info()
