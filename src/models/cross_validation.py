"""
낙상 감지 - 교차검증 (Cross-Validation)  [논문용]
=====================================================
사용법:
    python cross_validation.py

필요 파일 (같은 폴더):
    features.npz   <- feature_extraction.py 의 출력

필요 라이브러리:
    pip install numpy pandas scikit-learn xgboost matplotlib

출력:
    cv_results_file.csv     <- 파일 단위 교차검증 결과 표
    cv_results_subject.csv  <- 피험자 단위 교차검증 결과 표
    cv_plot.png             <- 모델별 평균+-표준편차 그래프

------------------------------------------------------------
왜 교차검증인가
  단일 분할(75/25)은 '한 번 나눈 운'에 결과가 좌우된다.
  교차검증은 데이터를 여러 방식으로 나눠 여러 번 평가한 뒤
  평균 +- 표준편차로 보고한다. 표준편차가 작을수록 안정적인 모델.

두 가지 교차검증을 모두 수행
  1) 파일 단위 (StratifiedGroupKFold)
       - 같은 녹화 파일이 학습/평가에 동시에 들어가는 누수 방지
       - 모든 동작이 평가에 포함됨
  2) 피험자 단위 (Leave-One-Subject-Out, LOSO)
       - 피험자 한 명을 통째로 빼고 평가 -> '처음 보는 사람'에 대한 일반화 측정
       - 낙상 감지 논문에서 가장 신뢰받는 평가 방식
------------------------------------------------------------
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score

RANDOM_STATE = 42
N_FOLDS = 5                  # 파일 단위 교차검증 fold 수


# ---------- 평가 지표 ----------
def fold_metrics(y_true, y_pred, y_proba):
    """한 fold 의 낙상 감지 지표."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1   = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    # ROC-AUC 는 한 fold 에 한 클래스만 있으면 계산 불가 -> 방어
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = np.nan
    return dict(sensitivity=sens, specificity=spec,
                precision=prec, f1=f1, roc_auc=auc)


_XGB_WARNED = False

# ---------- 모델 정의 ----------
def get_models(scale_pos_weight=1.0):
    global _XGB_WARNED
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
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE, eval_metric='logloss')
    except ImportError:
        if not _XGB_WARNED:
            print('[안내] xgboost 미설치 -> XGBoost 제외. '
                  '설치하려면: pip install xgboost')
            _XGB_WARNED = True
    return models


# ---------- 교차검증 실행 ----------
def run_cv(X, y, groups, splitter, split_args, models, cv_name):
    """splitter 로 데이터를 나눠가며 모든 모델을 평가.
    각 모델의 fold 별 지표를 모아 평균/표준편차로 요약."""
    print(f'\n{"="*60}\n{cv_name}\n{"="*60}')

    # fold 인덱스 미리 생성
    folds = list(splitter.split(X, y, groups, *split_args))
    print(f'  fold 수: {len(folds)}')

    rows = []
    for name, _ in models.items():
        per_fold = []
        for tr, te in folds:
            # 모델은 fold 마다 새로 만들어 학습 (누수 방지)
            model = get_models(
                scale_pos_weight=(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
            )[name]
            model.fit(X[tr], y[tr])
            pred  = model.predict(X[te])
            proba = model.predict_proba(X[te])[:, 1]
            per_fold.append(fold_metrics(y[te], pred, proba))

        # fold 평균 +- 표준편차
        summary = {'model': name}
        for metric in ['sensitivity', 'specificity', 'precision', 'f1', 'roc_auc']:
            vals = np.array([f[metric] for f in per_fold], dtype=float)
            summary[f'{metric}_mean'] = np.nanmean(vals)
            summary[f'{metric}_std']  = np.nanstd(vals)
        rows.append(summary)
        print(f'  {name:20s} F1 = {summary["f1_mean"]:.3f} '
              f'+- {summary["f1_std"]:.3f}')

    df = pd.DataFrame(rows).sort_values('f1_mean', ascending=False)
    return df


def main():
    d = np.load('features.npz', allow_pickle=True)
    X       = d['features']
    y       = d['y'].astype(int)
    source  = d['source']
    subject = np.array([str(s).lstrip('0') or '0' for s in d['subject']])

    print(f'데이터: {X.shape[0]}개 윈도우, {X.shape[1]}개 특징')
    print(f'  낙상 {int(y.sum())} / 비낙상 {int((y==0).sum())}')
    print(f'  피험자 {len(set(subject))}명: {sorted(set(subject))}')

    models = get_models()

    # ---------- 1) 파일 단위 교차검증 ----------
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)
    df_file = run_cv(X, y, source, sgkf, (), models,
                     f'1) 파일 단위 교차검증 ({N_FOLDS}-fold)')
    df_file.to_csv('cv_results_file.csv', index=False)

    # ---------- 2) 피험자 단위 교차검증 (LOSO) ----------
    n_subj = len(set(subject))
    if n_subj >= 3:
        loso = LeaveOneGroupOut()
        df_subj = run_cv(X, y, subject, loso, (), models,
                         f'2) 피험자 단위 교차검증 (Leave-One-Subject-Out, '
                         f'피험자 {n_subj}명)')
        df_subj.to_csv('cv_results_subject.csv', index=False)
    else:
        df_subj = None
        print('\n[안내] 피험자가 3명 미만이라 피험자 단위 교차검증은 생략.')

    # ---------- 결과 출력 ----------
    def show(df, title):
        print(f'\n----- {title} (F1 평균 내림차순) -----')
        out = df.copy()
        for m in ['sensitivity', 'specificity', 'f1', 'roc_auc']:
            out[m] = (out[f'{m}_mean'].round(3).astype(str) + ' ± '
                      + out[f'{m}_std'].round(3).astype(str))
        print(out[['model', 'sensitivity', 'specificity',
                   'f1', 'roc_auc']].to_string(index=False))

    show(df_file, '파일 단위 교차검증')
    if df_subj is not None:
        show(df_subj, '피험자 단위 교차검증 (처음 보는 사람 일반화)')

    # ---------- 그래프 ----------
    panels = [('File-level CV', df_file)]
    if df_subj is not None:
        panels.append(('Subject-level CV (LOSO)', df_subj))

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]
    metrics = ['sensitivity', 'specificity', 'f1']
    for ax, (title, df) in zip(axes, panels):
        names = list(df['model'])
        x = np.arange(len(metrics))
        w = 0.8 / len(names)
        for i, name in enumerate(names):
            means = [df.loc[df['model'] == name, f'{m}_mean'].values[0]
                     for m in metrics]
            errs  = [df.loc[df['model'] == name, f'{m}_std'].values[0]
                     for m in metrics]
            ax.bar(x + i * w, means, w, yerr=errs, capsize=3, label=name)
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.08)
        ax.set_title(title)
        ax.set_ylabel('score (mean ± std)')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig('cv_plot.png', dpi=120)
    print('\ncv_plot.png 저장')

    print('\n[해석 팁] 평균(F1)이 높아도 표준편차(±)가 크면 불안정한 모델이다.')
    print('          평균이 높고 ± 가 작은 모델이 가장 믿을 만하다.')
    print('          파일 단위보다 피험자 단위 점수가 낮은 것은 정상이며,')
    print('          이쪽이 실제 배포 성능에 더 가깝다.')


if __name__ == '__main__':
    main()
