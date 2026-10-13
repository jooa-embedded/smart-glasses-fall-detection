#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
낙상 감지 데이터셋 수집 스크립트
====================================
XIAO nRF52840 Sense가 USB 시리얼로 보내는 CSV 데이터를 받아서,
SPACE 키로 구간을 지정하면 그 구간만 개별 CSV 파일로 저장합니다.

[필요 패키지]
    pip install pyserial keyboard matplotlib

[조작법]
    SPACE : 녹화 시작 / 다시 누르면 종료 + 파일 저장
    f     : 다음 구간 라벨을 'fall' (낙상) 로 설정
    n     : 다음 구간 라벨을 'adl'  (일상동작) 로 설정
    q     : 프로그램 종료

[중요] 이 스크립트를 실행하는 동안에는 Arduino IDE의
       시리얼 모니터/플로터를 반드시 닫아야 합니다.
       (하나의 COM 포트는 한 프로그램만 사용할 수 있음)
"""

import os
import time
import threading
from datetime import datetime

import serial
import serial.tools.list_ports
import keyboard
import matplotlib
matplotlib.use("Agg")   # GUI 없이 파일로만 저장
import matplotlib.pyplot as plt


# ================= 설정 =================
PORT = None              # 예: "COM5". None이면 실행 시 목록에서 직접 선택
BAUD = 115200
SAVE_DIR = "recordings"  # 구간 파일이 저장될 최상위 폴더
# 낙상 라벨 목록 — 여기 있으면 fall/, 없으면 adl/ 에 저장
FALL_LABELS = {"fall", "fall_forward", "fall_backward", "fall_left", "fall_right"}
CSV_HEADER = "index,timestamp_ms,label,ax,ay,az,gx,gy,gz,accMag"
# Teleplot 신호 이름 순서 (CSV 열 순서와 일치해야 함)
TELEPLOT_SIGNALS = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ", "AccMag"]

# 세션 메타데이터 (main() 시작 시 입력)
SUBJECT_ID = ""
WEAR_POSITION = ""
TRIAL_NO = ""


# ================= 공유 상태 =================
recording = False        # 현재 녹화 중인지
buffer = []              # 녹화 중인 구간의 데이터 라인 저장
current_label = "fall"   # 다음/현재 구간 라벨
label_counts = {}        # 라벨별 저장 횟수 (파일 번호용)
latest_line = ""         # 가장 최근에 받은 라인 (화면 표시용)
running = True
lock = threading.Lock()
rec_start_dt = None      # 녹화 시작 시각 (취득 날짜 기록용)


def print_help():
    print("""
============= 낙상 데이터 수집 =============
  SPACE : 녹화 시작 / 종료(=파일 저장)
  f     : 다음 구간 라벨 = 'fall' (낙상)
  n     : 다음 구간 라벨 = 'adl'  (일상동작)
  q     : 종료
===========================================
""")


def serial_reader(ser):
    """백그라운드 스레드: 시리얼을 계속 읽어 녹화 중이면 버퍼에 쌓는다."""
    global latest_line
    while running:
        try:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
        except Exception:
            continue
        if not raw or raw.startswith("#") or raw.startswith("index,"):
            continue  # 빈 줄 / 주석(#) / 헤더 라인은 무시
        with lock:
            latest_line = raw
            if recording:
                buffer.append(raw)


def parse_teleplot(raw_lines):
    """Teleplot 라인들(>name:ts:value)을 timestamp 기준으로 묶어 CSV 행 리스트로 변환한다."""
    groups = {}   # {ts: {signal: value}}
    order = []    # timestamp 순서 보존
    for ln in raw_lines:
        if not ln.startswith(">"):
            continue
        parts = ln[1:].split(":")
        if len(parts) != 3:
            continue
        name, ts_str, val = parts
        try:
            ts = int(ts_str)
        except ValueError:
            continue
        if ts not in groups:
            groups[ts] = {}
            order.append(ts)
        groups[ts][name] = val.strip()

    rows = []
    idx = 0
    for ts in order:
        g = groups[ts]
        if not all(sig in g for sig in TELEPLOT_SIGNALS):
            continue  # 신호 일부가 없는 타임스탬프(녹화 첫/끝 경계)는 버림
        row = [str(idx), str(ts), current_label]
        for sig in TELEPLOT_SIGNALS:
            row.append(g[sig])
        rows.append(",".join(row))
        idx += 1
    return rows


def save_segment(lines, start_dt):
    """녹화된 한 구간을 Teleplot → CSV 변환 후 파일로 저장한다."""
    if not lines:
        print("\n[!] 데이터가 없어 저장하지 않았습니다.")
        return

    csv_rows = parse_teleplot(lines)
    if not csv_rows:
        print("\n[!] 파싱된 데이터가 없습니다. Teleplot 형식(>name:ts:value)인지 확인하세요.")
        return

    n = label_counts.get(current_label, 0) + 1
    label_counts[current_label] = n
    ts = start_dt.strftime("%Y%m%d_%H%M%S")
    category = "fall" if current_label.strip() in FALL_LABELS else "adl"
    save_dir = os.path.join(SAVE_DIR, category)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{current_label}_{n:03d}_{ts}.csv")

    # 실측 샘플링 레이트 계산
    sample_rate_str = "N/A"
    dur = 0.0
    try:
        t0 = int(csv_rows[0].split(",")[1])
        t1 = int(csv_rows[-1].split(",")[1])
        dur = (t1 - t0) / 1000.0
        if dur > 0:
            sample_rate_str = f"{len(csv_rows) / dur:.1f}"
    except Exception:
        pass

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# subject_id: {SUBJECT_ID}\n")
        f.write(f"# trial_no: {TRIAL_NO}\n")
        f.write(f"# label: {current_label}\n")
        f.write(f"# date: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# sample_rate_hz: {sample_rate_str}\n")
        f.write(f"# wear_position: {WEAR_POSITION}\n")
        f.write(CSV_HEADER + "\n")
        for row in csv_rows:
            f.write(row + "\n")

    info = f"{len(csv_rows)} samples"
    if dur > 0:
        info += f", {dur:.1f}s, ~{sample_rate_str} Hz"
    print(f"\n[저장 완료] {path}  ({info})")
    save_plot(csv_rows, path, start_dt)


def save_plot(csv_rows, path_csv, start_dt):
    """녹화 구간의 IMU 데이터를 2-패널 그래프로 저장한다."""
    # CSV 열: index(0) ts(1) label(2) ax(3) ay(4) az(5) gx(6) gy(7) gz(8) accMag(9)
    t, ax_l, ay_l, az_l, gx_l, gy_l, gz_l, mag_l = [], [], [], [], [], [], [], []
    for ln in csv_rows:
        p = ln.split(",")
        if len(p) < 10:
            continue
        try:
            ts_v  = int(p[1])
            ax_v  = float(p[3]); ay_v = float(p[4]); az_v = float(p[5])
            gx_v  = float(p[6]); gy_v = float(p[7]); gz_v = float(p[8])
            mag_v = float(p[9])
        except ValueError:
            continue  # 빈 값이 있는 행(녹화 첫 샘플 등)은 통째로 건너뜀
        t.append(ts_v)
        ax_l.append(ax_v); ay_l.append(ay_v); az_l.append(az_v)
        gx_l.append(gx_v); gy_l.append(gy_v); gz_l.append(gz_v)
        mag_l.append(mag_v)

    if not t:
        return

    t0 = t[0]
    sec = [(x - t0) / 1000.0 for x in t]

    fig, (top, bot) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(
        f"label={current_label}  |  {start_dt.strftime('%Y-%m-%d %H:%M:%S')}"
        f"  |  subject={SUBJECT_ID}  |  trial={TRIAL_NO}  |  pos={WEAR_POSITION}",
        fontsize=9
    )

    top.plot(sec, ax_l, label="ax")
    top.plot(sec, ay_l, label="ay")
    top.plot(sec, az_l, label="az")
    top.plot(sec, mag_l, label="accMag", linestyle="--", color="black")
    top.set_ylabel("Accel (g)")
    top.legend(loc="upper right", fontsize=8)
    top.grid(alpha=0.3)

    bot.plot(sec, gx_l, label="gx")
    bot.plot(sec, gy_l, label="gy")
    bot.plot(sec, gz_l, label="gz")
    bot.set_ylabel("Gyro (dps)")
    bot.set_xlabel("Time (s)")
    bot.legend(loc="upper right", fontsize=8)
    bot.grid(alpha=0.3)

    fig.tight_layout()
    path_png = path_csv[:-4] + ".png"
    fig.savefig(path_png, dpi=150)
    plt.close(fig)
    print(f"[그래프 저장] {path_png}")


_last_toggle_time = 0.0

def toggle_recording():
    """SPACE 키 콜백: 녹화 시작 / 종료 토글."""
    global recording, buffer, rec_start_dt, _last_toggle_time
    # 키 반복(key repeat) 방지: 마지막 토글로부터 0.5초 이내 호출은 무시
    now = time.time()
    if now - _last_toggle_time < 0.5:
        return
    _last_toggle_time = now

    lines = None
    start_dt = None
    with lock:
        if not recording:
            buffer = []
            recording = True
            rec_start_dt = datetime.now()
            print(f"\n[REC ●] 녹화 시작 (label='{current_label}')")
        else:
            recording = False
            lines = buffer[:]   # 복사본
            start_dt = rec_start_dt
            buffer = []
    if lines is not None:       # 저장은 lock 밖에서
        save_segment(lines, start_dt)


def set_label(label):
    """라벨 변경 (녹화 중에는 변경 불가)."""
    global current_label
    if recording:
        print("\n[!] 녹화 중에는 라벨을 바꿀 수 없습니다. SPACE로 먼저 종료하세요.")
        return
    current_label = label
    print(f"\n[라벨] 다음 구간 → '{current_label}'")


def ask_session_info():
    """프로그램 시작 시 피험자 ID, 기기 착용 위치, 시행번호를 입력받는다."""
    global SUBJECT_ID, WEAR_POSITION, TRIAL_NO
    print("\n=== 세션 정보 입력 ===")
    SUBJECT_ID = input("피험자 ID (예: S001): ").strip() or "unknown"
    print("착용 위치 예: waist(허리), wrist(손목), ankle(발목), chest(가슴)")
    WEAR_POSITION = input("기기 착용 위치: ").strip() or "unknown"
    TRIAL_NO = input("시행번호 (예: T01): ").strip() or "unknown"
    print(f"→ 피험자: {SUBJECT_ID}, 착용 위치: {WEAR_POSITION}, 시행번호: {TRIAL_NO}")


def pick_port():
    """연결된 시리얼 포트를 보여주고 사용자가 고르게 한다."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    print("사용 가능한 시리얼 포트:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device}  -  {p.description}")
    sel = input("XIAO가 연결된 포트 번호를 입력하세요: ").strip()
    try:
        return ports[int(sel)].device
    except Exception:
        return None


def main():
    global PORT, running

    os.makedirs(SAVE_DIR, exist_ok=True)
    ask_session_info()

    if PORT is None:
        PORT = pick_port()
    if PORT is None:
        print("포트를 찾을 수 없습니다. XIAO USB 연결을 확인하세요.")
        return

    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
    except Exception as e:
        print(f"포트 열기 실패: {e}")
        print("→ Arduino IDE의 시리얼 모니터가 열려 있지 않은지 확인하세요.")
        return

    time.sleep(2.0)            # 보드 리셋 안정화 대기
    ser.reset_input_buffer()
    print(f"\n연결됨: {PORT} @ {BAUD} baud")
    print_help()

    # 시리얼 읽기 스레드 시작
    threading.Thread(target=serial_reader, args=(ser,), daemon=True).start()

    # 키 바인딩 (라벨을 더 추가하려면 아래에 add_hotkey 줄을 늘리면 됨)
    keyboard.add_hotkey("space", toggle_recording)
    keyboard.add_hotkey("f", lambda: set_label("fall_left"))
    keyboard.add_hotkey("n", lambda: set_label("fall_right"))

    # 상태 표시 루프
    try:
        while running:
            with lock:
                status = "REC ●" if recording else "대기 "
                cnt = len(buffer)
                line = latest_line
            print(f"\r[{status}] label={current_label:6s} "
                  f"buffer={cnt:5d}  last: {line[:48]:48s}", end="")
            if keyboard.is_pressed("q"):
                running = False
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        if recording:               # 녹화 중 종료 시 마지막 구간 저장
            save_segment(buffer[:], rec_start_dt)
        ser.close()
        print("\n\n프로그램을 종료합니다.")


if __name__ == "__main__":
    main()
