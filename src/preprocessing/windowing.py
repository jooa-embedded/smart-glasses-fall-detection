"""
낙상 감지 - 윈도잉 파이프라인
================================
입력 : data/fall/*.csv, data/adl/*.csv  (메타데이터 6줄 + 신호)
출력 : windowed_dataset.npz  (X, y, action, subject, source)
       windowed_metadata.csv (윈도우별 메타정보)

처리 내용
  1. 각 파일을 정확히 100Hz 균일 격자로 리샘플링 -> 2.0초 = 200 샘플 고정
  2. 이산 동작(낙상, jump, sit 등): accMag 최대점(=핵심 이벤트)을 찾아
     그 지점이 충격 1.2초 전 ~ 0.8초 후가 되도록 윈도우 1개 추출
  3. 연속 동작(walk/run/static, 60초): 앞뒤 2초(전환구간)를 잘라내고
     2.0초 윈도우를 50% 겹쳐가며 슬라이딩
  4. 각 윈도우에 이진 클래스(fall=1/nonfall=0) + action_label + subject_id 부여

* 나중에 계단 데이터를 추가하면 data/ 폴더에 파일만 넣고 이 스크립트를
  다시 실행하면 됩니다.
"""
import os, re, glob
import numpy as np
import pandas as pd

# ---------------- 설정 ----------------
WINDOW_SEC      = 2.0          # 윈도우 길이 (초)
TARGET_HZ       = 100          # 리샘플링 목표 주파수
WIN_LEN         = int(WINDOW_SEC * TARGET_HZ)         # = 200 샘플
PRE_IMPACT_SEC  = 1.2          # 이산 동작: 이벤트 앞쪽으로 담을 길이
POST_IMPACT_SEC = 0.8          # 이산 동작: 이벤트 뒤쪽으로 담을 길이
TRIM_SEC        = 2.0          # 연속 동작: 앞뒤로 잘라낼 전환구간
OVERLAP         = 0.5          # 연속 동작: 윈도우 겹침 비율
CHANNELS        = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'accMag']

# 연속(주기적) 동작 = 60초 녹화. 그 외는 모두 이산 동작.
CONTINUOUS = {'walk_slow', 'walk_fast', 'run', 'sit_static', 'stand_static'}

DATA_DIRS = ['recordings/fall', 'recordings/adl']
OUT_NPZ   = 'windowed_dataset.npz'
OUT_CSV   = 'windowed_metadata.csv'

# 사용할 피험자 목록 (None 이면 전체 사용)
SUBJECT_FILTER = None   # 전체 피험자 사용


# ---------------- 파일 읽기 ----------------
def read_meta(path):
    """파일 상단의 '# key: value' 메타데이터 6줄을 딕셔너리로."""
    meta = {}
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            if line.startswith('#'):
                m = re.match(r'#\s*([^:]+):\s*(.*)', line.strip())
                if m:
                    meta[m.group(1).strip()] = m.group(2).strip()
            else:
                break
    return meta


def read_signal(path):
    """주석(#) 줄을 건너뛰고 신호 데이터프레임을 반환."""
    return pd.read_csv(path, comment='#', encoding='utf-8-sig')


def resample_100hz(df):
    """timestamp_ms 기준으로 정확히 100Hz 균일 격자에 선형보간.
    accMag는 보간 후 ax,ay,az로 다시 계산해 일관성을 유지."""
    t = df['timestamp_ms'].to_numpy(dtype=float)
    t = t - t[0]                                   # 0ms 부터 시작
    grid = np.arange(0, t[-1], 1000.0 / TARGET_HZ) # 10ms 간격
    out = {}
    for ch in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
        out[ch] = np.interp(grid, t, df[ch].to_numpy(dtype=float))
    out['accMag'] = np.sqrt(out['ax']**2 + out['ay']**2 + out['az']**2)
    return np.column_stack([out[ch] for ch in CHANNELS])   # (n_samples, 7)


# ---------------- 윈도우 추출 ----------------
def window_discrete(sig):
    """이산 동작: accMag 최대점을 이벤트로 보고 윈도우 1개 추출.
    파일이 2초보다 짧으면 가장자리 값으로 패딩."""
    acc = sig[:, CHANNELS.index('accMag')]
    event = int(np.argmax(acc))                    # 핵심 이벤트(낙상=충격) 위치
    start = event - int(PRE_IMPACT_SEC * TARGET_HZ)
    end   = start + WIN_LEN
    # 신호 범위를 벗어나면 가장자리 값으로 패딩
    pad_l = max(0, -start)
    pad_r = max(0, end - len(sig))
    s = max(0, start)
    e = min(len(sig), end)
    win = sig[s:e]
    if pad_l or pad_r:
        win = np.pad(win, ((pad_l, pad_r), (0, 0)), mode='edge')
    return [win[:WIN_LEN]]                         # 리스트로 반환(통일성)


def window_continuous(sig):
    """연속 동작: 앞뒤 전환구간을 잘라내고 50% 겹침 슬라이딩."""
    trim = int(TRIM_SEC * TARGET_HZ)
    core = sig[trim: len(sig) - trim]
    step = int(WIN_LEN * (1 - OVERLAP))            # = 100 샘플
    wins = []
    for start in range(0, len(core) - WIN_LEN + 1, step):
        wins.append(core[start:start + WIN_LEN])
    return wins


# ---------------- 메인 ----------------
def main():
    X, y, actions, subjects, sources = [], [], [], [], []

    files = []
    for d in DATA_DIRS:
        files += sorted(glob.glob(os.path.join(d, '*.csv')))
    print(f'총 {len(files)}개 파일 처리 시작...\n')

    for path in files:
        meta = read_meta(path)
        action = meta.get('label', '').strip()    # 내부 메타데이터의 라벨 사용(파일명 X)
        subject = meta.get('subject_id', '?')
        if SUBJECT_FILTER and subject not in SUBJECT_FILTER:
            continue
        df = read_signal(path)
        if len(df) < 10:
            print(f'  [건너뜀] 데이터 부족: {os.path.basename(path)}')
            continue
        sig = resample_100hz(df)

        if action in CONTINUOUS:
            wins = window_continuous(sig)
        else:
            wins = window_discrete(sig)

        cls = 1 if action.startswith('fall') else 0   # 이진 클래스
        for w in wins:
            if w.shape != (WIN_LEN, len(CHANNELS)):
                continue
            X.append(w)
            y.append(cls)
            actions.append(action)
            subjects.append(subject)
            sources.append(os.path.basename(path))

    X = np.asarray(X, dtype=np.float32)            # (n_windows, 200, 7)
    y = np.asarray(y, dtype=np.int8)
    actions = np.asarray(actions)
    subjects = np.asarray(subjects)
    sources = np.asarray(sources)

    np.savez_compressed(OUT_NPZ, X=X, y=y, action=actions,
                        subject=subjects, source=sources, channels=CHANNELS)
    pd.DataFrame({'class': y, 'action': actions,
                  'subject': subjects, 'source': sources}).to_csv(OUT_CSV, index=False)

    # ---- 요약 출력 ----
    print(f'완료: 윈도우 {len(X)}개,  형태 {X.shape}  (윈도우수, 샘플200, 채널7)\n')
    print('--- 클래스별 ---')
    print(f'  낙상(fall=1)    : {(y == 1).sum()}개')
    print(f'  비낙상(nonfall=0): {(y == 0).sum()}개')
    print('\n--- 동작별 윈도우 수 ---')
    for a in sorted(set(actions)):
        print(f'  {a:16s} {(actions == a).sum():5d}')
    print('\n--- 피험자별 ---')
    for s in sorted(set(subjects)):
        print(f'  subject {s}: {(subjects == s).sum()}개')


if __name__ == '__main__':
    main()
