"""
낙상 감지 - 여러 모델 비교 & 평가  (논문용)
=================================================
사용법:
    python compare_models.py

필요 파일 (같은 폴더):
    features.npz   <- feature_extraction.py 의 출력

출력:
    model_comparison.csv   <- 논문용 모델 비교 표 (단일 분할 기준)
    comparison_plot.png    <- 모델별 지표 막대그래프
    confusion_matrices.png <- 모델별 혼동행렬
    best_model.joblib      <- 최종 선택 모델(XGBoost) 저장   ★변경됨

------------------------------------------------------------
[ 모델 저장 정책 - 중요 ]
  이 표(model_comparison.csv)는 '단일 분할' 기준이므로, 여기서의 F1 1등이
  반드시 최종 모델은 아니다. 최종 모델 선택의 근거는
  cross_validation.py 의 '피험자 단위 교차검증'(처음 보는 사람 일반화)이며,
  그 결과 가장 높고 안정적인 모델은 XGBoost 이다.
  따라서 best_model.joblib 에는 단일 분할 1등(k-NN)이 아니라 XGBoost 를 저장한다.
  k-NN 은 학습에서 본 사람에는 강하나 처음 보는 사람에는 불안정하여
  최종 배포 모델로 부적합하다.
------------------------------------------------------------
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score
import joblib

RANDOM_STATE = 42

# ★변경: 최종 저장 모델을 한 곳에서 지정 (피험자 단위 교차검증 근거로 선택)
FINAL_MODEL_NAME = 'XGBoost'


def evaluate(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1   = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    acc  = (tp + tn) / (tp + tn + fp + fn)
    auc  = roc_auc_score(y_true, y_proba)
    return dict(accuracy=acc, sensitivity=sens, specificity=spec,
                precision=prec, f1=f1, roc_auc=auc,
                TP=tp, FN=fn, TN=tn, FP=fp)


def get_models():
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=300, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1),
        'SVM (RBF)': make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf', class_weight='balanced',
                probability=True, random_state=RANDOM_STATE)),
        'LogisticRegression': make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight='balanced',
                                max_iter=1000, random_state=RANDOM_STATE)),
        'k-NN': make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=7)),
        'LDA': LinearDiscriminantAnalysis(),
    }
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            random_state=RANDOM_STATE, eval_metric='logloss')
    except ImportError:
        print('[안내] xgboost 미설치 -> XGBoost 제외. '
              '설치하려면: pip install xgboost')
    return models


def main():
    d = np.load('features.npz', allow_pickle=True)
    X      = d['features']
    y      = d['y'].astype(int)
    source = d['source']
    print(f'데이터: {X.shape[0]}개 윈도우, {X.shape[1]}개 특징')
    print(f'  낙상 {int(y.sum())} / 비낙상 {int((y==0).sum())}\n')

    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True,
                                random_state=RANDOM_STATE)
    tr_idx, te_idx = next(sgkf.split(X, y, groups=source))
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    leak = set(source[tr_idx]) & set(source[te_idx])
    print(f'학습 {len(tr_idx)} / 테스트 {len(te_idx)}  '
          f'(겹친 파일 {len(leak)}개, 0이면 정상)\n')

    models = get_models()
    if 'XGBoost' in models:
        models['XGBoost'].set_params(
            scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1))

    results = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred  = model.predict(Xte)
        proba = model.predict_proba(Xte)[:, 1]
        results[name] = evaluate(yte, pred, proba)
        print(f'  {name:20s} 완료')

    df = pd.DataFrame(results).T
    df = df[['sensitivity', 'specificity', 'precision', 'f1',
             'roc_auc', 'accuracy', 'TP', 'FN', 'TN', 'FP']]
    df = df.sort_values('f1', ascending=False)
    df.to_csv('model_comparison.csv')
    pd.set_option('display.width', 120)
    print('\n===== 모델 비교 결과 (단일 분할, F1 내림차순) =====')
    print(df.round(3).to_string())

    single_best = df.index[0]
    print(f'\n[참고] 단일 분할 F1 1등: {single_best} '
          f'(F1={df.loc[single_best,"f1"]:.3f}) - 저장 대상 아님')

    # ★변경: 단일 분할 1등이 아니라 피험자 단위 CV로 선택한 최종 모델(XGBoost) 저장
    if FINAL_MODEL_NAME in models:
        save_name = FINAL_MODEL_NAME
    else:
        save_name = single_best
        print(f'[경고] {FINAL_MODEL_NAME} 미설치 -> 차선책으로 {save_name} 저장')
    joblib.dump(models[save_name], 'best_model.joblib')
    print(f'\n>>> best_model.joblib 에 [{save_name}] 저장 완료 '
          f'(피험자 단위 교차검증 근거)')

    metrics = ['sensitivity', 'specificity', 'precision', 'f1', 'roc_auc']
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(metrics))
    w = 0.8 / len(df)
    for i, name in enumerate(df.index):
        ax.bar(x + i * w, [df.loc[name, m] for m in metrics], w, label=name)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('score')
    ax.set_title('Model comparison - fall detection metrics')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig('comparison_plot.png', dpi=120)
    print('\ncomparison_plot.png 저장')

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6))
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, df.index):
        r = results[name]
        cm = np.array([[r['TN'], r['FP']], [r['FN'], r['TP']]])
        ax.imshow(cm, cmap='Blues')
        for ii in range(2):
            for jj in range(2):
                ax.text(jj, ii, cm[ii, jj], ha='center', va='center',
                        fontsize=13,
                        color='white' if cm[ii, jj] > cm.max() / 2 else 'black')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['pred\nnonfall', 'pred\nfall'], fontsize=8)
        ax.set_yticklabels(['true\nnonfall', 'true\nfall'], fontsize=8)
        ax.set_title(f'{name}\nsens={r["sensitivity"]:.2f} '
                     f'spec={r["specificity"]:.2f}', fontsize=9)
    fig.tight_layout()
    fig.savefig('confusion_matrices.png', dpi=120)
    print('confusion_matrices.png 저장')
    print('\n완료. model_comparison.csv 를 논문 표로 사용하세요.')


if __name__ == '__main__':
    main()