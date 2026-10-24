"""
실시간 낙상 검출 실험 분석

integrated_monitor.py의 윈도우 로그(vitals_log_YYYY-MM-DD.csv)와
experiment_marker.py의 마커(markers_YYYY-MM-DD.csv)를 시간으로 맞춰,
각 낙상 시도 구간 안에서 낙상 분류 점수 최댓값이 0.5 이상이면 검출로 판정한다.
결과는 experiment_result_YYYY-MM-DD.csv로 저장하고 방향별 검출률을 출력한다.

실행: python3 analyze_experiment.py [YYYY-MM-DD]   (날짜 생략 시 오늘)
"""
import csv
import datetime
import sys
from collections import defaultdict

DETECTION_THRESHOLD = 0.5
UNCLOSED_TRIAL_SEC = 30     # 종료 마커가 없는 시도는 시작 후 30초로 간주

DIR_KR = {'forward': '전방', 'backward': '후방', 'left': '좌측', 'right': '우측'}


def parse_dt(s):
    s = s.strip()
    try:
        return datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        return datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')


def load_markers(date_str):
    """마커 CSV를 읽어 시도별 시작·종료 시각을 묶는다."""
    fn = f"markers_{date_str}.csv"
    trials, open_trials = [], {}
    try:
        with open(fn, 'r') as f:
            for row in csv.DictReader(f):
                tid = row['trial_id']
                ts = parse_dt(row['timestamp'])
                if row['event'] == 'start':
                    open_trials[tid] = {
                        'trial_id': tid,
                        'direction': row['direction'],
                        'start': ts,
                        'end': None,
                    }
                elif row['event'] == 'end' and tid in open_trials:
                    open_trials[tid]['end'] = ts
                    open_trials[tid]['end_note'] = row.get('note', '')
                    trials.append(open_trials.pop(tid))
    except FileNotFoundError:
        print(f"마커 파일 없음: {fn}")
        sys.exit(1)

    for trial in open_trials.values():
        trial['end'] = trial['start'] + datetime.timedelta(seconds=UNCLOSED_TRIAL_SEC)
        trial['end_note'] = 'no_end_marker'
        trials.append(trial)

    trials.sort(key=lambda t: t['start'])
    return trials


def load_vitals(date_str):
    """모니터링 로그에서 윈도우별 시각·낙상 점수·등급을 읽는다."""
    fn = f"vitals_log_{date_str}.csv"
    rows = []
    try:
        with open(fn, 'r') as f:
            for row in csv.DictReader(f):
                try:
                    rows.append({
                        'ts': parse_dt(row['Timestamp']),
                        'fall_prob': float(row.get('FallProb', 0)),
                        'activity': row.get('Activity', ''),
                        'grade': row.get('Grade', ''),
                    })
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        print(f"모니터링 로그 파일 없음: {fn}")
        sys.exit(1)
    return rows


def analyze(trials, vitals):
    """각 시도 구간 안의 윈도우 중 낙상 점수 최댓값으로 검출 여부를 판정."""
    results = []
    for trial in trials:
        in_range = [v for v in vitals if trial['start'] <= v['ts'] <= trial['end']]
        if in_range:
            max_window = max(in_range, key=lambda v: v['fall_prob'])
            max_prob = max_window['fall_prob']
            grade_at_max = max_window['grade']
            window_grades = sorted({v['grade'] for v in in_range if v['grade']})
        else:
            max_prob, grade_at_max, window_grades = 0.0, "", []

        results.append({
            'trial_id': trial['trial_id'],
            'direction': trial['direction'],
            'start': trial['start'].strftime('%H:%M:%S.%f')[:-3],
            'end': trial['end'].strftime('%H:%M:%S.%f')[:-3],
            'duration_s': (trial['end'] - trial['start']).total_seconds(),
            'n_windows': len(in_range),
            'max_fall_prob': max_prob,
            'detected': max_prob >= DETECTION_THRESHOLD,
            'grade_at_max': grade_at_max,
            'observed_grades': '|'.join(window_grades),
            'end_note': trial.get('end_note', ''),
        })
    return results


def save_results(results, date_str):
    fn = f"experiment_result_{date_str}.csv"
    with open(fn, 'w', newline='') as f:
        if not results:
            return fn
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        for r in results:
            row = r.copy()
            row['max_fall_prob'] = f"{row['max_fall_prob']:.3f}"
            row['detected'] = 'TRUE' if row['detected'] else 'FALSE'
            row['duration_s'] = f"{row['duration_s']:.1f}"
            writer.writerow(row)
    return fn


def print_summary(results):
    """방향별 검출률(낙상 시도 중 검출된 비율)을 출력."""
    by_dir = defaultdict(list)
    for r in results:
        by_dir[r['direction']].append(r)

    print()
    print("=" * 70)
    print("  실시간 낙상 검출 결과")
    print("=" * 70)
    print(f"\n  검출 기준: 시도 구간 내 낙상 점수 최댓값 ≥ {DETECTION_THRESHOLD}\n")
    print(f"  {'방향':6} {'시도수':>5} {'검출':>5} {'실패':>5} {'검출률':>8} "
          f"{'평균점수':>10} {'최대점수':>10}")
    print("  " + "─" * 60)

    total_n = total_detected = 0
    all_probs = []
    for direction in ['forward', 'backward', 'left', 'right']:
        if direction not in by_dir:
            continue
        trials = by_dir[direction]
        n = len(trials)
        detected = sum(1 for t in trials if t['detected'])
        probs = [t['max_fall_prob'] for t in trials]
        print(f"  {DIR_KR[direction]:6} {n:>5} {detected:>5} {n - detected:>5} "
              f"{detected / n * 100:>7.1f}% {sum(probs) / n:>9.3f} {max(probs):>9.3f}")
        total_n += n
        total_detected += detected
        all_probs.extend(probs)

    print("  " + "─" * 60)
    if total_n:
        print(f"  {'전체':6} {total_n:>5} {total_detected:>5} {total_n - total_detected:>5} "
              f"{total_detected / total_n * 100:>7.1f}% {sum(all_probs) / total_n:>9.3f}")
    print()

    failed = [r for r in results if not r['detected']]
    if failed:
        print(f"  검출 실패 시도 ({len(failed)}건):")
        for t in failed:
            d_kr = DIR_KR.get(t['direction'], t['direction'])
            print(f"    - {d_kr} {t['trial_id']}: 최대 점수={t['max_fall_prob']:.3f}, "
                  f"윈도우 {t['n_windows']}개")
        print()


def main():
    date_str = sys.argv[1] if len(sys.argv) > 1 else datetime.datetime.now().strftime('%Y-%m-%d')
    print(f"분석 날짜: {date_str}\n")

    trials = load_markers(date_str)
    print(f"  마커: {len(trials)}개 시도")
    vitals = load_vitals(date_str)
    print(f"  모니터링 로그: {len(vitals)}개 윈도우")

    results = analyze(trials, vitals)
    print_summary(results)
    print(f"  상세 결과 저장: {save_results(results, date_str)}\n")


if __name__ == "__main__":
    main()
