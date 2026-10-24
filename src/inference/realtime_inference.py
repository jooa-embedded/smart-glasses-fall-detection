"""
IMU 실시간 추론 모듈 (Real-time Inference)
================================================
학습 때 만든 두 모델을 불러와, 실시간으로 들어오는 IMU 신호에 적용한다.
  - 낙상 감지 모델   : best_model.joblib      (낙상 / 비낙상)
  - 활동 분류 모델   : activity_model.joblib  (운동 active / 안정 resting)

[ 학습과 추론의 일관성 - 매우 중요 ]
  추론은 학습 때와 '완전히 똑같은' 방식으로 신호를 처리해야 한다.
  - 같은 채널 순서: ax, ay, az, gx, gy, gz, accMag
  - 같은 윈도우 크기: 200 샘플 (100Hz x 2초)
  - 같은 특징 추출: feature_extraction.extract_window
  하나라도 다르면 모델이 엉뚱한 값을 받아 오작동한다.

사용 개념
  1) IMURealtimeBuffer: 들어오는 IMU 샘플을 하나씩 넣으면, 200개가
     모일 때마다 윈도우를 반환한다 (이후 150샘플마다 새 윈도우).
  2) FallActivityInference: 윈도우 1개를 받아 낙상/활동을 예측한다.

  실제 사용 흐름:
     buf = IMURealtimeBuffer()
     inf = FallActivityInference()
     # 센서 샘플이 들어올 때마다:
     win = buf.add(ax, ay, az, gx, gy, gz)   # 윈도우 안 차면 None
     if win is not None:
         result = inf.predict(win)
         # result = {'fall': True/False, 'fall_prob': ...,
         #           'activity': 'active'/'resting', ...}
"""
import numpy as np
import joblib

from feature_extraction import extract_window

CHANNELS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'accMag']
WINDOW_LEN = 200          # 100Hz x 2.0초 (학습과 동일)
SLIDE = 150               # 150샘플마다 새 윈도우 (50샘플 겹침)


# ---------------------------------------------------------------
# 1) 실시간 IMU 버퍼
# ---------------------------------------------------------------
class IMURealtimeBuffer:
    """IMU 샘플을 하나씩 받아 쌓다가, WINDOW_LEN개가 모이면 윈도우를 반환.
    이후 SLIDE개를 비우고 계속 채운다 (슬라이딩 윈도우).

    accMag(가속도 크기)는 학습 때와 동일하게 ax,ay,az로부터 계산한다.
    """
    def __init__(self, window_len=WINDOW_LEN, slide=SLIDE):
        self.window_len = window_len
        self.slide = slide
        self.buf = []          # 각 원소: [ax,ay,az,gx,gy,gz,accMag]

    def add(self, ax, ay, az, gx, gy, gz):
        """샘플 1개 추가. 윈도우가 가득 차면 (window_len, 7) 배열 반환, 아니면 None."""
        acc_mag = float(np.sqrt(ax*ax + ay*ay + az*az))
        self.buf.append([ax, ay, az, gx, gy, gz, acc_mag])
        if len(self.buf) >= self.window_len:
            window = np.array(self.buf[:self.window_len], dtype=np.float32)
            del self.buf[:self.slide]      # 앞 slide개 비우기 (오버랩 유지)
            return window
        return None

    def reset(self):
        self.buf.clear()


# ---------------------------------------------------------------
# 2) 낙상 + 활동 추론기
# ---------------------------------------------------------------
class FallActivityInference:
    """두 모델을 불러와 윈도우 하나에 대해 낙상/활동을 예측."""

    def __init__(self, fall_model_path='best_model.joblib',
                 activity_model_path='activity_model.joblib'):
        self.fall_model = joblib.load(fall_model_path)
        act = joblib.load(activity_model_path)
        # activity_model.joblib 은 dict로 저장됨 (model + 라벨정보)
        self.activity_model = act['model'] if isinstance(act, dict) else act

    def _features(self, window):
        """윈도우(200x7) -> 특징 벡터(1 x 115). 학습과 동일한 함수 사용."""
        feat = list(extract_window(window).values())
        F = np.array([feat], dtype=np.float32)
        return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    def _prob(self, model, F):
        """모델이 predict_proba를 지원하면 양성 확률, 아니면 예측값을 확률처럼 반환."""
        if hasattr(model, 'predict_proba'):
            return float(model.predict_proba(F)[0, 1])
        return float(model.predict(F)[0])

    def predict(self, window):
        """윈도우 1개에 대해 낙상/활동 예측 결과 딕셔너리 반환."""
        F = self._features(window)
        fall_prob = self._prob(self.fall_model, F)
        act_prob = self._prob(self.activity_model, F)
        return {
            'fall':        fall_prob >= 0.5,
            'fall_prob':   round(fall_prob, 3),
            'activity':    'active' if act_prob >= 0.5 else 'resting',
            'activity_prob': round(act_prob, 3),
        }


# ---------------------------------------------------------------
# 3) 단독 실행 시: 저장된 윈도우로 추론이 맞는지 검증
# ---------------------------------------------------------------
if __name__ == '__main__':
    print('=== 추론 모듈 검증 (저장된 윈도우로 모의 실시간 테스트) ===\n')
    d = np.load('windowed_dataset.npz', allow_pickle=True)
    X, y, action = d['X'], d['y'].astype(int), d['action']

    inf = FallActivityInference()

    # 각 클래스에서 몇 개 골라 예측이 맞는지 확인
    import random
    random.seed(0)
    test_idx = (list(np.where(y == 1)[0][:5]) +      # 낙상 5개
                list(np.where(action == 'walk_fast')[0][:3]) +
                list(np.where(action == 'sit_static')[0][:3]) +
                list(np.where(action == 'run')[0][:2]))

    print(f"{'action':14s} {'true':8s} {'pred_fall':10s} {'fall_p':7s} {'activity':9s}")
    print('-' * 56)
    correct = 0
    for i in test_idx:
        r = inf.predict(X[i])
        true_fall = 'fall' if y[i] == 1 else 'nonfall'
        pred_fall = 'fall' if r['fall'] else 'nonfall'
        ok = '✓' if true_fall == pred_fall else '✗'
        if true_fall == pred_fall:
            correct += 1
        print(f"{action[i]:14s} {true_fall:8s} {pred_fall:10s} "
              f"{r['fall_prob']:<7} {r['activity']:9s} {ok}")
    print(f"\n낙상 판정 정확: {correct}/{len(test_idx)}")

    # 버퍼 동작 검증: 샘플을 하나씩 흘려넣어 윈도우가 만들어지는지
    print('\n=== 버퍼 동작 검증 (샘플 스트림 모의) ===')
    buf = IMURealtimeBuffer()
    sample_win = X[np.where(y == 1)[0][0]]   # 낙상 윈도우 하나를 스트림처럼 흘림
    made = 0
    for row in sample_win:
        w = buf.add(row[0], row[1], row[2], row[3], row[4], row[5])
        if w is not None:
            made += 1
            r = inf.predict(w)
            print(f"  윈도우 생성됨 -> 예측: fall={r['fall']} "
                  f"(p={r['fall_prob']}), activity={r['activity']}")
    print(f"  200개 샘플 중 생성된 윈도우 수: {made} (200개째에 1개 생성이 정상)")
