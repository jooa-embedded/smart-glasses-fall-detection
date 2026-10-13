"""
활동 분류 모델 (운동 active / 안정 resting)
================================================
사용법:
    python activity_model.py

필요 파일 (같은 폴더):
    windowed_dataset.npz   <- windowing.py 의 출력
    feature_extraction.py  <- 특징 추출 함수 재사용

필요 라이브러리:
    pip install numpy scikit-learn xgboost joblib

출력:
    activity_model.joblib   <- 학습된 활동 분류 모델
    activity_features.npz   <- 활동 분류용 특징 데이터셋

------------------------------------------------------------
목적
  위급 판단 로직에서 "지금 운동 중인가, 안정 중인가"를 판정한다.
  심박 이상(빈맥 등)이 운동 때문인지 위급인지 구별하는 데 필요하다.
  기존의 임의 임계값(가속도 1.1G) 대신 머신러닝으로 판정하여 근거를 확보한다.

동작 분류 기준
  active  (운동): walk_slow, walk_fast, run, up_stair, down_stair, jump
  resting (안정): sit_static, stand_static, lie_down, pick_up,
                  squat, stumble, fast_sit, stand_up
  * 낙상(fall_*)은 별개 사건이므로 활동 분류에서 제외한다.
------------------------------------------------------------
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.metrics import confusion_matrix
import joblib

from feature_extraction import extract_window

RANDOM_STATE = 42

# 동작 -> 활동 라벨 매핑
ACTIVE  = {'walk_slow', 'walk_fast', 'run', 'up_stair', 'down_stair', 'jump'}
RESTING = {'sit_static', 'stand_static', 'lie_down', 'pick_up',
           'squat', 'stumble', 'fast_sit', 'stand_up'}
# active=1, resting=0


def build_model(scale_pos_weight=1.0):
    """활동 분류 모델. 낙상에서 XGBoost가 가장 좋았으므로 동일 계열 사용.
    xgboost 미설치 시 RandomForest로 대체."""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                             scale_pos_weight=scale_pos_weight,
                             random_state=RANDOM_STATE, eval_metric='logloss')
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                      random_state=RANDOM_STATE, n_jobs=-1)


def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    # active(1) 기준 재현율/정밀도
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return acc, f1


def main():
    d = np.load('windowed_dataset.npz', allow_pickle=True)
    Xw      = d['X']
    action  = d['action']
    source  = d['source']
    subject = np.array([str(s).lstrip('0') or '0' for s in d['subject']])

    # 활동 라벨 부여 (낙상은 제외)
    keep, y = [], []
    for i, a in enumerate(action):
        if a in ACTIVE:
            keep.append(i); y.append(1)
        elif a in RESTING:
            keep.append(i); y.append(0)
        # fall_* 는 건너뜀
    keep = np.array(keep); y = np.array(y, dtype=int)
    Xw_k, src_k, subj_k, act_k = Xw[keep], source[keep], subject[keep], action[keep]

    print(f'활동 분류 대상 윈도우: {len(keep)}개')
    print(f'  active(운동) {int((y==1).sum())} / resting(안정) {int((y==0).sum())}')
    print(f'  피험자 {len(set(subj_k))}명\n')

    # 특징 추출
    print('특징 추출 중...')
    F = np.array([list(extract_window(w).values()) for w in Xw_k], dtype=np.float32)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    # ----- 파일 단위 교차검증 -----
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    for tr, te in sgkf.split(F, y, groups=src_k):
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = build_model(spw); m.fit(F[tr], y[tr])
        a, f = metrics(y[te], m.predict(F[te]))
        accs.append(a); f1s.append(f)
    print('--- 파일 단위 5-fold 교차검증 ---')
    print(f'  정확도 {np.mean(accs):.3f} ± {np.std(accs):.3f}')
    print(f'  F1     {np.mean(f1s):.3f} ± {np.std(f1s):.3f}')

    # ----- 피험자 단위 교차검증 -----
    if len(set(subj_k)) >= 3:
        logo = LeaveOneGroupOut()
        accs2, f1s2 = [], []
        for tr, te in logo.split(F, y, groups=subj_k):
            spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
            m = build_model(spw); m.fit(F[tr], y[tr])
            a, f = metrics(y[te], m.predict(F[te]))
            accs2.append(a); f1s2.append(f)
        print('--- 피험자 단위 교차검증 (처음 보는 사람) ---')
        print(f'  정확도 {np.mean(accs2):.3f} ± {np.std(accs2):.3f}')
        print(f'  F1     {np.mean(f1s2):.3f} ± {np.std(f1s2):.3f}')

    # ----- 전체 데이터로 최종 모델 학습 & 저장 -----
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    final = build_model(spw); final.fit(F, y)
    joblib.dump({'model': final,
                 'active_labels': sorted(ACTIVE),
                 'resting_labels': sorted(RESTING)},
                'activity_model.joblib')
    np.savez_compressed('activity_features.npz',
                        features=F, y=y, action=act_k, subject=subj_k)
    print('\n저장: activity_model.joblib, activity_features.npz')
    print('사용법: 모델 예측이 1이면 active(운동 중), 0이면 resting(안정 중)')


if __name__ == '__main__':
    main()
