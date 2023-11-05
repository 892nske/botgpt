import numpy as np

# 短期および長期の単純移動平均 (SMA) を計算する関数
def calculate_sma(dataframe, column, window):
    return dataframe[column].rolling(window=window).mean()

# ボリンジャーバンドの上限、下限、中心線を計算する関数
def calculate_bollinger_bands(dataframe, column, window, num_std_dev):
    sma = dataframe[column].rolling(window=window).mean()
    std_dev = dataframe[column].rolling(window=window).std()
    
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    
    return upper_band, lower_band, sma


# RSIを計算する関数
def calculate_rsi(dataframe, column, window):
    delta = dataframe[column].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# MACDおよびシグナルラインを計算する関数
def calculate_macd(dataframe, column, short_window, long_window, signal_window):
    short_ema = dataframe[column].ewm(span=short_window).mean()
    long_ema = dataframe[column].ewm(span=long_window).mean()

    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window).mean()

    return macd, signal_line

# ヒストリカルボラティリティを計算する関数
def calculate_historical_volatility(dataframe, column, window):
    log_returns = np.log(dataframe[column] / dataframe[column].shift(1))
    historical_volatility = log_returns.rolling(window=window).std() * np.sqrt(252) * 100

    return historical_volatility


# フィボナッチリトレースメントを計算する関数
def calculate_fibonacci_retracements(high, low):
    retracement_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    retracements = {}
    
    for level in retracement_levels:
        retracements[level] = high - (high - low) * level

    return retracements


# ストキャスティクスを計算する関数
def calculate_stochastics(dataframe, high_col, low_col, close_col, k_window, d_window):
    highest_high = dataframe[high_col].rolling(window=k_window).max()
    lowest_low = dataframe[low_col].rolling(window=k_window).min()
    
    stoch_k = (dataframe[close_col] - lowest_low) / (highest_high - lowest_low) * 100
    stoch_d = stoch_k.rolling(window=d_window).mean()

    return stoch_k, stoch_d


# オンバランスボリュームを計算する関数
def calculate_obv(dataframe, close_col, volume_col):
    obv = [0]
    
    for i in range(1, len(dataframe)):
        if dataframe[close_col][i] > dataframe[close_col][i-1]:
            obv.append(obv[-1] + dataframe[volume_col][i])
        elif dataframe[close_col][i] < dataframe[close_col][i-1]:
            obv.append(obv[-1] - dataframe[volume_col][i])
        else:
            obv.append(obv[-1])
            
    return obv
