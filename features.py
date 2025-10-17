import numpy as np
def feats_window(X):  # X: (T, 6) ax..gz
    f = []
    for c in range(X.shape[1]):
        x = X[:,c]
        f += [x.mean(), x.std(), x.min(), x.max(), np.ptp(x),
              np.percentile(x,25), np.percentile(x,75)]
        # 频域
        fx = np.abs(np.fft.rfft(x - x.mean()))
        f += [fx.mean(), fx.std(), fx.argmax()/len(fx)]
    # 形态：过零率、峰度、偏度
    from scipy.stats import kurtosis, skew
    for c in range(X.shape[1]):
        x = X[:,c]; f += [((x[:-1]*x[1:])<0).mean(), kurtosis(x), skew(x)]
    return np.array(f, dtype=np.float32)
