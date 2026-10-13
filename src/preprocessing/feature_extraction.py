"""
낙상 감지 - 특징 추출 파이프라인
==================================
입력 : windowed_dataset.npz   (X: 윈도우 1229 x 200샘플 x 7채널)
출력 : features.csv           (윈도우별 특징 한 줄 + 라벨)
       features.npz           (모델용 배열 형태)

각 윈도우(200x7)에서 채널별로 시간영역 / 주파수영역 특징을 뽑아
'숫자 한 줄'로 요약합니다. 1229개 윈도우 -> 1229행짜리 표.

* 윈도잉과 마찬가지로, 데이터가 늘어나면 windowed_dataset.npz만 새로
  만든 뒤 이 스크립트를 다시 돌리면 됩니다.
"""
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

FS        = 100                       # 샘플링 주파수 (Hz)
CHANNELS  = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'accMag']
IN_NPZ    = 'windowed_dataset.npz'
OUT_CSV   = 'features.csv'
OUT_NPZ   = 'features.npz'


def time_features(x):
    """한 채널 신호(200샘플)에서 시간영역 특징."""
    mean = x.mean()
    return {
        'mean'   : mean,
        'std'    : x.std(),
        'min'    : x.min(),
        'max'    : x.max(),
        'range'  : x.max() - x.min(),
        'rms'    : np.sqrt(np.mean(x**2)),
        'mad'    : np.mean(np.abs(x - mean)),          # 평균절대편차
        'iqr'    : np.percentile(x, 75) - np.percentile(x, 25),
        'skew'   : skew(x),                            # 비대칭도
        'kurt'   : kurtosis(x),                        # 첨도
        'energy' : np.sum(x**2) / len(x),
        'mcr'    : np.mean(np.abs(np.diff(np.sign(x - mean)))) / 2,  # 평균교차율
    }


def freq_features(x):
    """한 채널 신호에서 주파수영역 특징 (FFT 기반)."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    mag = np.abs(np.fft.rfft(x - x.mean()))            # DC 성분 제거
    power = mag**2
    total = power.sum() + 1e-12
    p = power / total                                  # 정규화 파워 스펙트럼
    dom_freq = freqs[np.argmax(mag)]                   # 우세 주파수
    spec_entropy = -np.sum(p * np.log(p + 1e-12))      # 스펙트럼 엔트로피
    mean_freq = np.sum(freqs * power) / total          # 평균 주파수
    return {
        'domfreq' : dom_freq,
        'specent' : spec_entropy,
        'meanfreq': mean_freq,
        'specenergy': power.sum() / n,
    }


def extract_window(win):
    """윈도우 1개(200x7) -> 특징 딕셔너리 하나."""
    feat = {}
    for c, ch in enumerate(CHANNELS):
        x = win[:, c].astype(float)
        for k, v in time_features(x).items():
            feat[f'{ch}_{k}'] = v
        for k, v in freq_features(x).items():
            feat[f'{ch}_{k}'] = v
    # 낙상 특화 보조 특징
    acc = win[:, CHANNELS.index('accMag')].astype(float)
    jerk = np.diff(acc) * FS                           # 가속도 변화율
    feat['accMag_peakjerk'] = np.max(np.abs(jerk))     # 충격의 급격함
    feat['accMag_p2p']      = acc.max() - acc.min()    # 자유낙하-충격 폭
    # 가속도 3축 신호크기면적(SMA)
    sma = np.mean(np.abs(win[:, 0]) + np.abs(win[:, 1]) + np.abs(win[:, 2]))
    feat['acc_sma'] = sma
    return feat


def main():
    d = np.load(IN_NPZ, allow_pickle=True)
    X = d['X']
    print(f'입력: 윈도우 {len(X)}개  형태 {X.shape}\n')

    rows = [extract_window(w) for w in X]
    feat_df = pd.DataFrame(rows)

    # NaN/inf 방어 (상수 신호 등에서 발생 가능)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 라벨 정보 결합
    feat_df.insert(0, 'class',   d['y'])
    feat_df.insert(1, 'action',  d['action'])
    feat_df.insert(2, 'subject', d['subject'])
    feat_df.insert(3, 'source',  d['source'])

    feat_df.to_csv(OUT_CSV, index=False)

    feat_cols = [c for c in feat_df.columns
                 if c not in ('class', 'action', 'subject', 'source')]
    np.savez_compressed(
        OUT_NPZ,
        features=feat_df[feat_cols].to_numpy(dtype=np.float32),
        feature_names=np.array(feat_cols),
        y=d['y'], action=d['action'],
        subject=d['subject'], source=d['source'])

    # ---- 요약 ----
    print(f'완료: {len(feat_df)}행 x {len(feat_cols)}개 특징\n')
    print(f'채널 {len(CHANNELS)}개 x (시간 12 + 주파수 4) + 보조 3 = {len(feat_cols)}개\n')
    print('--- 특징 예시 (낙상 vs 비낙상 평균 비교) ---')
    sample = ['accMag_max', 'accMag_min', 'accMag_peakjerk',
              'accMag_p2p', 'acc_sma', 'gx_std', 'az_meanfreq']
    fall = feat_df[feat_df['class'] == 1]
    non  = feat_df[feat_df['class'] == 0]
    print(f'{"feature":18s} {"낙상 평균":>12s} {"비낙상 평균":>12s}')
    for f in sample:
        print(f'{f:18s} {fall[f].mean():12.3f} {non[f].mean():12.3f}')


if __name__ == '__main__':
    main()
