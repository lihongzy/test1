import pandas as pd


def load_data() -> pd.DataFrame:
    # Load CSV with common encodings.
    path = r"data\[张永平]XiaMen2024-共享单车、电单车.csv"
    df = None
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk", "big5"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            print("encoding:", enc)
            break
        except Exception:
            pass
    if df is None:
        raise SystemExit("读取失败")
    return df
