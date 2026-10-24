"""
낙상 실험 마커 도구

integrated_monitor.py가 실행 중일 때 별도 터미널에서 실행해,
각 낙상 시도의 시작·종료 시각을 markers_YYYY-MM-DD.csv에 기록한다.
기록된 마커는 analyze_experiment.py에서 시도별 검출 여부를 계산하는 데 쓴다.

키: 1 전방 / 2 후방 / 3 좌측 / 4 우측 낙상 시작, e 시도 종료,
    s 진행 상태, h 도움말, q 실험 종료
시작 후 30초가 지나거나 새 시도를 시작하면 이전 시도는 자동 종료된다.

실행: python3 experiment_marker.py
"""
import csv
import datetime
import os
import threading
import time

AUTO_CLOSE_SEC = 30
MAX_TRIALS_PER_DIRECTION = 10

DIRECTIONS = {'1': 'forward', '2': 'backward', '3': 'left', '4': 'right'}
DIR_KR = {'forward': '전방', 'backward': '후방', 'left': '좌측', 'right': '우측'}

current_trial = None
trial_counts = {d: 0 for d in DIRECTIONS.values()}
marker_lock = threading.Lock()


# ===== 로그 파일 =====
def get_log_filename():
    return f"markers_{datetime.datetime.now():%Y-%m-%d}.csv"


def init_log():
    fn = get_log_filename()
    if not os.path.exists(fn):
        with open(fn, 'w', newline='') as f:
            csv.writer(f).writerow(['trial_id', 'direction', 'event', 'timestamp', 'note'])
    return fn


def log_event(trial_id, direction, event, note=""):
    with open(get_log_filename(), 'a', newline='') as f:
        csv.writer(f).writerow([
            trial_id,
            direction,
            event,
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            note,
        ])


# ===== 화면 =====
def show_status():
    print()
    print("=" * 56)
    print("  진행 상황")
    print("=" * 56)
    for d_key, d_eng in DIRECTIONS.items():
        count = trial_counts[d_eng]
        bar = "▓" * count + "░" * (MAX_TRIALS_PER_DIRECTION - count)
        done = " (완료)" if count == MAX_TRIALS_PER_DIRECTION else ""
        print(f"  [{d_key}] {DIR_KR[d_eng]:6s} {bar} {count}/{MAX_TRIALS_PER_DIRECTION}{done}")
    print("=" * 56)
    if current_trial:
        elapsed = time.time() - current_trial['start_time']
        d_kr = DIR_KR[current_trial['direction']]
        print(f"  진행 중: {d_kr} 시도 {current_trial['trial_id']} ({elapsed:.1f}초 경과)")
    else:
        print("  대기 중 (키를 눌러 시작)")
    print()


def show_help():
    print()
    print("  낙상 실험 마커 도구")
    print("  " + "-" * 40)
    print("    1 : 전방 낙상 시작")
    print("    2 : 후방 낙상 시작")
    print("    3 : 좌측 낙상 시작")
    print("    4 : 우측 낙상 시작")
    print("    e : 현재 시도 종료")
    print("    s : 진행 상태 확인")
    print("    q : 실험 전체 종료")
    print("  " + "-" * 40)
    print("  방향 키 → 동작 수행 → e 키로 종료")
    print(f"  시작 후 {AUTO_CLOSE_SEC}초가 지나면 자동 종료")
    print()


# ===== 시도 제어 =====
def start_trial(direction):
    global current_trial
    with marker_lock:
        if current_trial:
            d_prev_kr = DIR_KR[current_trial['direction']]
            print(f"  이전 시도({d_prev_kr} {current_trial['trial_id']}) 자동 종료")
            log_event(current_trial['trial_id'], current_trial['direction'],
                      'end', 'auto_close_new_trial')
            current_trial = None

        trial_counts[direction] += 1
        trial_id = f"{direction}_{trial_counts[direction]:02d}"
        current_trial = {
            'direction': direction,
            'trial_id': trial_id,
            'start_time': time.time(),
        }
        log_event(trial_id, direction, 'start')
        print(f"  {DIR_KR[direction]} 시도 {trial_counts[direction]}/{MAX_TRIALS_PER_DIRECTION} "
              f"시작 → 동작 수행 후 'e' 키로 종료")


def end_trial(reason="manual"):
    global current_trial
    with marker_lock:
        if not current_trial:
            print("  진행 중인 시도가 없습니다.")
            return
        elapsed = time.time() - current_trial['start_time']
        d_kr = DIR_KR[current_trial['direction']]
        log_event(current_trial['trial_id'], current_trial['direction'],
                  'end', f"{reason}_{elapsed:.1f}s")
        print(f"  {d_kr} 시도 {current_trial['trial_id']} 종료 ({elapsed:.1f}초)")
        current_trial = None


def auto_close_watcher():
    while True:
        time.sleep(1)
        if current_trial and time.time() - current_trial['start_time'] > AUTO_CLOSE_SEC:
            print(f"\n  {AUTO_CLOSE_SEC}초 경과 - 자동 종료")
            end_trial(reason="auto_timeout")


# ===== 키 입력 =====
def handle_key(key):
    key = key.strip().lower()
    if not key:
        return True

    if key in DIRECTIONS:
        direction = DIRECTIONS[key]
        if trial_counts[direction] >= MAX_TRIALS_PER_DIRECTION:
            print(f"  {DIR_KR[direction]} 낙상은 이미 {MAX_TRIALS_PER_DIRECTION}회 완료. "
                  f"다음으로 진행하세요.")
            return True
        start_trial(direction)
    elif key == 'e':
        end_trial()
    elif key == 's':
        show_status()
    elif key == 'q':
        if current_trial:
            end_trial(reason="experiment_quit")
        print("\n  실험 종료. 데이터 저장 파일:", get_log_filename())
        show_status()
        return False
    elif key in ('h', '?'):
        show_help()
    else:
        print(f"  알 수 없는 키: '{key}' (h 입력 시 도움말)")
    return True


def main():
    fn = init_log()
    show_help()
    print(f"  마커 저장 파일: {fn}\n")

    threading.Thread(target=auto_close_watcher, daemon=True).start()

    while True:
        try:
            key = input("> ")
        except (EOFError, KeyboardInterrupt):
            if current_trial:
                end_trial(reason="ctrl_c")
            print("\n  실험 종료.")
            break
        if not handle_key(key):
            break


if __name__ == "__main__":
    main()
