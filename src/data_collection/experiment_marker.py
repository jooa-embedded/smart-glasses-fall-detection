# “””
낙상 실험 마커 도구

별도 터미널에서 실행. 메인 시스템(integrated_monitor_ble_v10.py)이
돌고 있는 상태에서 이 스크립트로 실험 마커를 찍는다.

[ 키 입력 ]
1 : 전방 낙상 시작
2 : 후방 낙상 시작
3 : 좌측 낙상 시작
4 : 우측 낙상 시작
e : 현재 시도 종료
s : 현재 진행 상태 확인
q : 실험 전체 종료

[ 안전장치 ]

- 이전 시도 종료 안 했는데 새 시작 키 누르면: 이전 시도 자동 종료
- 시작 후 30초 경과: 자동 종료
- 키 입력 잊어도 데이터 안 망가짐

[ 출력 ]
markers_YYYY-MM-DD.csv 파일에 모든 마커 자동 저장
Columns: trial_id, direction, event(start/end), timestamp, auto_close_reason

[ 실험 흐름 예시 ]
1 → [전방 낙상 동작] → e   (전방 1회차)
1 → [전방 낙상 동작] → e   (전방 2회차)
… (10회 반복)
2 → [후방 낙상] → e        (후방 시작)
…
q (실험 종료)

실행: python3 experiment_marker.py
“””
import sys
import os
import csv
import time
import datetime
import threading

# ===== 설정 =====

AUTO_CLOSE_SEC = 30                # 시작 후 자동 종료까지 시간
MAX_TRIALS_PER_DIRECTION = 10      # 방향별 시도 수 (안내용)

DIRECTIONS = {
‘1’: ‘forward’,    # 전방
‘2’: ‘backward’,   # 후방
‘3’: ‘left’,       # 좌측
‘4’: ‘right’,      # 우측
}
DIR_KR = {
‘forward’: ‘전방’,
‘backward’: ‘후방’,
‘left’: ‘좌측’,
‘right’: ‘우측’,
}

# ===== 상태 변수 =====

current_trial = None      # {‘direction’:, ‘trial_id’:, ‘start_time’:, …}
trial_counts = {d: 0 for d in DIRECTIONS.values()}
marker_lock = threading.Lock()

# ===== 로그 파일 =====

def get_log_filename():
return f”markers_{datetime.datetime.now():%Y-%m-%d}.csv”

def init_log():
fn = get_log_filename()
if not os.path.exists(fn):
with open(fn, ‘w’, newline=’’) as f:
w = csv.writer(f)
w.writerow([‘trial_id’, ‘direction’, ‘event’, ‘timestamp’, ‘note’])
return fn

def log_event(trial_id, direction, event, note=””):
fn = get_log_filename()
with open(fn, ‘a’, newline=’’) as f:
w = csv.writer(f)
w.writerow([
trial_id,
direction,
event,
datetime.datetime.now().strftime(’%Y-%m-%d %H:%M:%S.%f’)[:-3],
note
])

# ===== 화면 =====

def show_status():
print()
print(”=” * 56)
print(”  📊 진행 상황”)
print(”=” * 56)
for d_key, d_eng in DIRECTIONS.items():
d_kr = DIR_KR[d_eng]
count = trial_counts[d_eng]
bar = “▓” * count + “░” * (MAX_TRIALS_PER_DIRECTION - count)
marker = “ ⭐” if count == MAX_TRIALS_PER_DIRECTION else “”
print(f”  [{d_key}] {d_kr:6s} {bar} {count}/{MAX_TRIALS_PER_DIRECTION}{marker}”)
print(”=” * 56)
if current_trial:
elapsed = time.time() - current_trial[‘start_time’]
d_kr = DIR_KR[current_trial[‘direction’]]
print(f”  🔴 진행 중: {d_kr} 시도 {current_trial[‘trial_id’]} ({elapsed:.1f}초 경과)”)
else:
print(f”  ⚪ 대기 중 (키를 눌러 시작)”)
print()

def show_help():
print()
print(“┌” + “─” * 54 + “┐”)
print(“│” + “  🔬 낙상 실험 마커 도구”.ljust(54) + “│”)
print(“├” + “─” * 54 + “┤”)
print(“│” + “  키 입력 안내”.ljust(54) + “│”)
print(“│” + “    1 : 전방 낙상 시작”.ljust(54) + “│”)
print(“│” + “    2 : 후방 낙상 시작”.ljust(54) + “│”)
print(“│” + “    3 : 좌측 낙상 시작”.ljust(54) + “│”)
print(“│” + “    4 : 우측 낙상 시작”.ljust(54) + “│”)
print(“│” + “    e : 현재 시도 종료”.ljust(54) + “│”)
print(“│” + “    s : 진행 상태 확인”.ljust(54) + “│”)
print(“│” + “    q : 실험 전체 종료”.ljust(54) + “│”)
print(“├” + “─” * 54 + “┤”)
print(“│” + “  방향 키 누름 → 동작 수행 → e 키로 종료”.ljust(54) + “│”)
print(f”│” + f”  안전장치: 시작 후 {AUTO_CLOSE_SEC}초 지나면 자동 종료”.ljust(54) + “│”)
print(“└” + “─” * 54 + “┘”)
print()

# ===== 트라이얼 제어 =====

def start_trial(direction):
global current_trial
with marker_lock:
# 이전 시도가 안 끝났으면 자동 종료
if current_trial:
d_prev_kr = DIR_KR[current_trial[‘direction’]]
print(f”  ⚠️  이전 시도({d_prev_kr} {current_trial[‘trial_id’]}) 자동 종료”)
log_event(current_trial[‘trial_id’], current_trial[‘direction’],
‘end’, ‘auto_close_new_trial’)
current_trial = None

```
    # 새 시도 시작
    trial_counts[direction] += 1
    trial_id = f"{direction}_{trial_counts[direction]:02d}"
    current_trial = {
        'direction': direction,
        'trial_id': trial_id,
        'start_time': time.time(),
    }
    log_event(trial_id, direction, 'start')
    d_kr = DIR_KR[direction]
    print(f"  ▶️  {d_kr} 시도 {trial_counts[direction]}/{MAX_TRIALS_PER_DIRECTION} 시작 → 동작 수행 후 'e' 키로 종료")
```

def end_trial(reason=“manual”):
global current_trial
with marker_lock:
if not current_trial:
print(”  ⚠️  진행 중인 시도가 없습니다.”)
return
elapsed = time.time() - current_trial[‘start_time’]
d_kr = DIR_KR[current_trial[‘direction’]]
log_event(current_trial[‘trial_id’], current_trial[‘direction’],
‘end’, f”{reason}_{elapsed:.1f}s”)
print(f”  ⏹️  {d_kr} 시도 {current_trial[‘trial_id’]} 종료 ({elapsed:.1f}초)”)
current_trial = None

# ===== 자동 종료 감시 스레드 =====

def auto_close_watcher():
while True:
time.sleep(1)
if current_trial:
elapsed = time.time() - current_trial[‘start_time’]
if elapsed > AUTO_CLOSE_SEC:
print(f”\n  ⏰ {AUTO_CLOSE_SEC}초 경과 - 자동 종료”)
end_trial(reason=“auto_timeout”)

# ===== 키 입력 처리 =====

def handle_key(key):
key = key.strip().lower()
if not key:
return True

```
if key in DIRECTIONS:
    direction = DIRECTIONS[key]
    if trial_counts[direction] >= MAX_TRIALS_PER_DIRECTION:
        d_kr = DIR_KR[direction]
        print(f"  ℹ️  {d_kr} 낙상은 이미 {MAX_TRIALS_PER_DIRECTION}회 완료. 다음으로 진행하세요.")
        return True
    start_trial(direction)
elif key == 'e':
    end_trial()
elif key == 's':
    show_status()
elif key == 'q':
    if current_trial:
        end_trial(reason="experiment_quit")
    print("\n  🏁 실험 종료. 데이터는", get_log_filename(), "에 저장되었습니다.")
    show_status()
    return False
elif key == 'h' or key == '?':
    show_help()
else:
    print(f"  ❓ 알 수 없는 키: '{key}' (h 입력 시 도움말)")
return True
```

def main():
fn = init_log()
show_help()
print(f”  💾 마커 저장 파일: {fn}\n”)

```
# 자동 종료 스레드
t = threading.Thread(target=auto_close_watcher, daemon=True)
t.start()

# 메인 입력 루프
while True:
    try:
        key = input("> ")
    except (EOFError, KeyboardInterrupt):
        if current_trial:
            end_trial(reason="ctrl_c")
        print("\n  🏁 실험 종료.")
        break
    
    if not handle_key(key):
        break
```

if **name** == “**main**”:
main()