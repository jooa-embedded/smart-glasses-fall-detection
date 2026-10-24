"""
스마트 안경 통합 모니터링 (라즈베리파이 5)

XIAO nRF52840에서 BLE로 IMU·PPG·EOG 신호를 받아
낙상/활동 분류, 생체정보 계산, 4단계 위험 등급 판단을 수행하고,
WARNING 이상이면 Gemini API로 상황 보고서를 생성한다.
API 호출이 실패하거나 15초 안에 응답이 없으면 고정 템플릿 보고서로 대체한다.

실행: GOOGLE_API_KEY=... python3 integrated_monitor.py
필요 파일 (같은 폴더): best_model.joblib, activity_model.joblib,
                     feature_extraction.py, realtime_inference.py
"""
import asyncio
import csv
import datetime
import os
import threading
import time
import warnings

import numpy as np
import heartpy as hp
import google.generativeai as genai
from bleak import BleakScanner, BleakClient

from realtime_inference import IMURealtimeBuffer, FallActivityInference

# heartpy 비수렴 경고 억제 (결과에는 영향 없음)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ===== Gemini API =====
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
model = genai.GenerativeModel('gemini-pro')
API_TIMEOUT_SEC = 15        # 응답이 없으면 고정 템플릿으로 전환
AI_LOCK_RELEASE_SEC = 5     # 보고서 생성 후 재호출 잠금 시간

# ===== BLE =====
XIAO_NAME = "XIAO SENSOR"
DATA_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"
EOG_ENABLED = True          # True: 9필드(IMU 6 + IR + RED + EOG), False: 8필드

# ===== PPG =====
BUFFER_SIZE = 400
SLIDE_SIZE = 150
HISTORY_SIZE_BPM = 5
HISTORY_SIZE_SLOW = 15

# ===== 생체정보 기준값 =====
HR_MAX_RESTING = 100        # 안정 시 빈맥 기준
HR_MIN_RESTING = 60         # 안정 시 서맥 기준
RESP_MAX = 20
RESP_MIN = 12
SPO2_SUSPECT = 95           # 안정 시 저산소 의심
SPO2_DANGER = 90            # 활동 중에도 위험으로 보는 기준

# ===== 등급 판단 시간 기준 =====
PERSISTENCE_WINDOWS = 3     # CAUTION이 연속 3개 윈도우 유지되면 WARNING
FALL_FOLLOWUP_SEC = 60      # 낙상 후 활동 복귀 관찰 시간
CANCEL_GRACE_SEC = 30       # CRITICAL 발송 전 취소 유예 시간

# ===== EOG 깜빡임 검출 =====
EOG_BUFFER_SIZE = 6000                # 100 Hz x 60 s
EOG_CALIB_SAMPLES = 3000              # 시작 후 30초간 기준선 측정
EOG_THRESHOLD_K = 2.0                 # 임계값 = mu + k * sigma
EOG_BLINK_REFRACTORY_SAMPLES = 30     # 불응기 0.3초
EOG_THRESHOLD_MIN = 50
EOG_THRESHOLD_MAX = 800
EOG_SIGMA_MIN = 5
EOG_FALLBACK_THRESHOLD = 200          # 캘리브레이션 실패 시 기본값
EOG_LOW_BLINK_THRESHOLD = 1           # 30초간 1회 미만이면 안구 활동 저하
EOG_COUNT_WINDOW_SAMPLES = 3000

# ===== 상태 변수 =====
bpm_history, spo2_history, rpm_history = [], [], []
persistence_counter = 0
fall_followup_until = 0
cancel_grace_until = 0
last_critical_trigger = ""
is_ai_analyzing = False

ir_buffer, red_buffer, timestamps = [], [], []
ppg_processing = False

eog_buffer = []
eog_blink_amplitude = None
eog_calib_done = False
eog_calib_samples = []
eog_blink_events = []
eog_last_blink_idx = -EOG_BLINK_REFRACTORY_SAMPLES
eog_sample_idx = 0

imu_buffer = IMURealtimeBuffer()
inference = FallActivityInference('best_model.joblib', 'activity_model.joblib')
imu_result = {'fall': False, 'fall_prob': 0.0, 'activity': 'resting'}
vitals = {'bpm': None, 'spo2': None, 'rpm': None}


# ===== 로그 =====
def get_daily_filename():
    return f"vitals_log_{datetime.datetime.now():%Y-%m-%d}.csv"


def log_to_csv(bpm, spo2, rpm, fall_prob, activity, grade, reason):
    fn = get_daily_filename()
    exists = os.path.exists(fn)
    with open(fn, 'a', newline='') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['Timestamp', 'BPM', 'SpO2', 'RPM', 'FallProb',
                        'Activity', 'Grade', 'Reason'])
        bpm_s = f"{bpm:.1f}" if bpm is not None else ""
        spo2_s = f"{spo2:.1f}" if spo2 is not None else ""
        rpm_s = f"{rpm:.1f}" if rpm is not None else ""
        w.writerow([f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
                    bpm_s, spo2_s, rpm_s, f"{fall_prob:.2f}",
                    activity, grade, reason])


def get_recent_logs(n=10):
    fn = get_daily_filename()
    try:
        if not os.path.exists(fn):
            return "이전 로그 없음"
        with open(fn) as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return "데이터 읽기 오류"


# ===== 보고서 =====
def generate_template_report(reason, grade):
    """Gemini API를 쓸 수 없을 때 사용하는 고정 템플릿 보고서."""
    bpm, spo2, resp = vitals['bpm'], vitals['spo2'], vitals['rpm']
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    activity = imu_result['activity']
    fall_p = imu_result['fall_prob']

    bpm_s = f"{bpm:.1f}" if bpm is not None else "측정 불가"
    spo2_s = f"{spo2:.1f}%" if spo2 is not None else "측정 불가"
    resp_s = f"{resp:.1f}회/분" if resp is not None else "측정 불가"
    urgency = "긴급" if grade == "CRITICAL" else "경고"
    ems_note = ("이미 발송되었거나 곧 발송됩니다" if grade == "CRITICAL"
                else "필요할 수 있습니다")

    return f"""
================================================================
[자동 생성 응급 보고서 - 고정 템플릿]
(인터넷 또는 AI API 사용 불가로 자동 생성됨)

▌ 발생 시각: {now}
▌ 위험 등급: {grade} ({urgency})
▌ 감지 사유: {reason}

▌ 측정값
- 심박          : {bpm_s} BPM
- 산소포화도    : {spo2_s}
- 호흡수        : {resp_s}
- 활동 상태     : {activity}
- 낙상 분류 점수: {fall_p:.2f}

▌ 보호자 안내
사용자의 스마트 안경에서 {urgency} 상황이 감지되었습니다.
즉시 사용자의 안전을 확인해 주십시오.
{urgency} 등급에 따라 응급 의료 서비스(119) 연락이 {ems_note}.

▌ 응급 센터(119) 브리핑 (참고용)
- 측정값: HR {bpm_s}, SpO2 {spo2_s}, RR {resp_s}
- 활동 상태: {activity}, 낙상 분류 점수: {fall_p:.2f}
- 위험 등급: {grade}

▌ 시스템 메모
AI 보고서 생성에 실패하여 고정 템플릿으로 대체되었습니다.
인터넷 연결 또는 API 상태를 점검해 주십시오.
================================================================
"""


async def request_emergency_report(reason, grade, recent_data):
    """Gemini API로 보고서를 요청하고, 실패하거나 시간 초과 시 고정 템플릿으로 대체."""
    global is_ai_analyzing
    is_ai_analyzing = True

    prompt = f"""
당신은 응급 의료 관제 시스템의 AI입니다.
사용자의 스마트 안경에서 다음 상황이 감지되었습니다.

[확정 등급]: {grade}  (이미 확정. 재판단 말고 그대로 전달)
[사유]: {reason}

[최근 윈도우 데이터 로그]:
{recent_data}

위 데이터로 다음 3대상 보고서를 한국어로 작성:
1. 보호자: 상황 요약 및 안내
2. 응급 센터(119): 측정값 기반 브리핑
3. 시스템 관리자: 센서 신뢰도
"""

    api_success = False
    try:
        print(f"\n\n[Gemini API 호출] 등급: {grade}")
        # 블로킹 호출을 스레드로 넘겨 BLE 수신이 멈추지 않게 함
        response = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, prompt),
            timeout=API_TIMEOUT_SEC,
        )
        print("\n" + "=" * 60)
        print(f"AI 보고서 생성 완료 ({grade})")
        print(response.text)
        print("=" * 60 + "\n")
        api_success = True
    except asyncio.TimeoutError:
        print(f"\nAPI 응답 지연 ({API_TIMEOUT_SEC}초 초과)")
    except Exception as e:
        print(f"\nAPI 호출 실패: {e}")

    if not api_success:
        print("[고정 템플릿 보고서로 전환]")
        template = generate_template_report(reason, grade)
        print(template)
        try:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"emergency_report_{ts}.txt", 'w', encoding='utf-8') as f:
                f.write(template)
        except OSError:
            pass

    await asyncio.sleep(AI_LOCK_RELEASE_SEC)
    is_ai_analyzing = False


# ===== PPG: SpO2 추정 =====
def estimate_spo2(ir_list, red_list, sample_rate):
    """Red/IR 비율(R)로 SpO2 추정. 신호 품질이 낮으면 None."""
    ir_data = np.asarray(ir_list, dtype=np.float64)
    red_data = np.asarray(red_list, dtype=np.float64)
    valid = (np.isfinite(ir_data) & np.isfinite(red_data) &
             (ir_data > 0) & (red_data > 0))
    ir_data, red_data = ir_data[valid], red_data[valid]
    if len(ir_data) < 100 or sample_rate <= 5:
        return None

    ir_dc, red_dc = np.median(ir_data), np.median(red_data)
    if ir_dc <= 0 or red_dc <= 0:
        return None

    high_cutoff = min(4.0, sample_rate * 0.45)
    if high_cutoff <= 0.7:
        return None
    ir_f = hp.filter_signal(ir_data, cutoff=[0.7, high_cutoff],
                            sample_rate=sample_rate, order=2, filtertype='bandpass')
    red_f = hp.filter_signal(red_data, cutoff=[0.7, high_cutoff],
                             sample_rate=sample_rate, order=2, filtertype='bandpass')

    ir_ac = np.sqrt(np.mean((ir_f - np.mean(ir_f)) ** 2))
    red_ac = np.sqrt(np.mean((red_f - np.mean(red_f)) ** 2))
    if ir_ac <= 0 or red_ac <= 0:
        return None

    corr = np.corrcoef(ir_f, red_f)[0, 1]
    if not np.isfinite(corr) or abs(corr) < 0.5:
        return None

    r_ratio = (red_ac / red_dc) / (ir_ac / ir_dc)
    if not np.isfinite(r_ratio) or r_ratio < 0.35 or r_ratio > 1.40:
        return None
    raw = -45.060 * (r_ratio ** 2) + 30.354 * r_ratio + 94.845
    return max(70.0, min(100.0, raw))


# ===== PPG: 심박·호흡·SpO2 =====
def calculate_vitals(ir_list, red_list, actual_sample_rate):
    global ppg_processing
    try:
        if len(ir_list) < 100 or len(red_list) < 100 or actual_sample_rate < 10:
            return
        ir_data = np.array(ir_list, dtype=np.float64)
        if not np.all(np.isfinite(ir_data)):
            return

        filtered_ir = hp.filter_signal(ir_data, cutoff=[0.7, 3.0],
                                       sample_rate=actual_sample_rate,
                                       order=2, filtertype='bandpass')
        wd, m = hp.process(filtered_ir, sample_rate=actual_sample_rate)
        raw_bpm = m['bpm']
        raw_rpm = m.get('breathingrate', 0.25) * 60

        if 8 <= raw_rpm <= 30:
            rpm_history.append(raw_rpm)
            if len(rpm_history) > HISTORY_SIZE_SLOW:
                rpm_history.pop(0)
        if rpm_history:
            vitals['rpm'] = sum(rpm_history) / len(rpm_history)

        raw_spo2 = estimate_spo2(ir_list, red_list, actual_sample_rate)
        if raw_spo2 is not None and 70 <= raw_spo2 <= 100:
            spo2_history.append(raw_spo2)
            if len(spo2_history) > HISTORY_SIZE_SLOW:
                spo2_history.pop(0)
        if spo2_history:
            vitals['spo2'] = min(100.0, sum(spo2_history) / len(spo2_history))

        if 40 <= raw_bpm <= 200:
            bpm_history.append(raw_bpm)
            if len(bpm_history) > HISTORY_SIZE_BPM:
                bpm_history.pop(0)
        if bpm_history:
            vitals['bpm'] = sum(bpm_history) / len(bpm_history)
    except Exception:
        # heartpy가 수렴하지 못한 윈도우는 건너뛰고 다음 윈도우에서 재시도
        pass
    finally:
        ppg_processing = False


def run_calculate_vitals_async(ir_list, red_list, rate):
    """PPG 처리를 별도 스레드에서 실행해 BLE 콜백이 막히지 않게 함.
    이전 처리가 끝나지 않았으면 이번 호출은 건너뜀."""
    global ppg_processing
    if ppg_processing:
        return
    ppg_processing = True
    th = threading.Thread(target=calculate_vitals,
                          args=(list(ir_list), list(red_list), rate),
                          daemon=True)
    th.start()


# ===== EOG: 깜빡임 검출 =====
def detect_blink(new_sample_value, current_sample_idx):
    """기준선 대비 진폭이 임계값을 넘으면 깜빡임으로 기록 (불응기 적용)."""
    global eog_last_blink_idx, eog_blink_events

    if not eog_calib_done or eog_blink_amplitude is None:
        return False
    if len(eog_buffer) < 100:
        return False

    baseline = np.mean(eog_buffer[-100:])
    amplitude = abs(new_sample_value - baseline)

    if current_sample_idx - eog_last_blink_idx < EOG_BLINK_REFRACTORY_SAMPLES:
        return False

    if amplitude > eog_blink_amplitude:
        eog_last_blink_idx = current_sample_idx
        eog_blink_events.append(time.time())
        cutoff = time.time() - 60
        eog_blink_events = [t for t in eog_blink_events if t > cutoff]
        return True
    return False


def finish_eog_calibration():
    """시작 후 30초 데이터로 깜빡임 임계값(mu + k*sigma)을 정하고 범위를 제한."""
    global eog_blink_amplitude, eog_calib_done

    if len(eog_calib_samples) < 500:
        eog_blink_amplitude = EOG_FALLBACK_THRESHOLD
        eog_calib_done = True
        return

    samples = np.array(eog_calib_samples)
    amplitudes = np.abs(samples - np.mean(samples))
    mu, sigma = np.mean(amplitudes), np.std(amplitudes)
    raw_threshold = mu + EOG_THRESHOLD_K * sigma

    if sigma < EOG_SIGMA_MIN:
        eog_blink_amplitude = EOG_FALLBACK_THRESHOLD
    elif raw_threshold < EOG_THRESHOLD_MIN:
        eog_blink_amplitude = EOG_THRESHOLD_MIN
    elif raw_threshold > EOG_THRESHOLD_MAX:
        eog_blink_amplitude = EOG_THRESHOLD_MAX
    else:
        eog_blink_amplitude = raw_threshold
    eog_calib_done = True


def get_blink_count_30s():
    cutoff = time.time() - 30
    return sum(1 for t in eog_blink_events if t > cutoff)


def is_eog_low_activity():
    """최근 30초 깜빡임이 기준 미만이면 True. 판단할 데이터가 부족하면 False."""
    if not eog_calib_done:
        return False
    if eog_sample_idx < EOG_CALIB_SAMPLES + EOG_COUNT_WINDOW_SAMPLES:
        return False
    return get_blink_count_30s() < EOG_LOW_BLINK_THRESHOLD


# ===== 위험 등급 판단 =====
def build_reason(prefix, fall, f_hr, f_spo2, f_resp, bpm, spo2, resp):
    parts = []
    if fall:
        parts.append("낙상")
    if f_hr and bpm is not None:
        if bpm > HR_MAX_RESTING:
            parts.append(f"빈맥({bpm:.0f}BPM)")
        elif bpm < HR_MIN_RESTING:
            parts.append(f"서맥({bpm:.0f}BPM)")
    if f_spo2 and spo2 is not None:
        parts.append(f"저산소({spo2:.1f}%)")
    if f_resp and resp is not None:
        parts.append(f"호흡이상({resp:.1f}RPM)")
    return f"{prefix}: " + ", ".join(parts) if parts else prefix


def evaluate_grade():
    """낙상 분류 결과와 생체정보 플래그를 조합해 등급을 결정."""
    bpm, spo2, resp = vitals['bpm'], vitals['spo2'], vitals['rpm']
    activity = imu_result['activity']
    fall_prob = imu_result['fall_prob']

    # 활동 상태에 따라 생체정보 기준을 다르게 적용
    if activity == 'resting':
        flag_hr = bpm is not None and (bpm > HR_MAX_RESTING or bpm < HR_MIN_RESTING)
        flag_resp = resp is not None and not (RESP_MIN <= resp <= RESP_MAX)
        flag_spo2 = spo2 is not None and spo2 < SPO2_SUSPECT
    else:
        flag_hr = False
        flag_resp = False
        flag_spo2 = spo2 is not None and spo2 < SPO2_DANGER

    # EOG 단독으로는 등급을 올리지 않음
    flag_eog = EOG_ENABLED and is_eog_low_activity()
    flag_fall = imu_result['fall']
    n_vital = int(flag_hr) + int(flag_resp) + int(flag_spo2)

    if flag_eog and (n_vital >= 1 or flag_fall or
                     (activity == 'resting' and (flag_hr or flag_spo2))):
        grade = 'CRITICAL'
        reason = build_reason("안구 활동 저하", flag_fall, flag_hr, flag_spo2,
                              flag_resp, bpm, spo2, resp)
    elif flag_fall and n_vital >= 1:
        grade = 'WARNING'
        reason = build_reason("생체 이상 동반 낙상", True, flag_hr, flag_spo2,
                              flag_resp, bpm, spo2, resp)
    elif flag_eog and n_vital >= 1:
        grade = 'WARNING'
        reason = build_reason("안구 활동 저하 + 생체 이상", False, flag_hr, flag_spo2,
                              flag_resp, bpm, spo2, resp)
    elif flag_fall:
        grade = 'CAUTION_FALL'
        reason = f"낙상 감지 (분류 점수 {fall_prob:.2f}, 60초 후속 관찰)"
    elif n_vital >= 1:
        grade = 'CAUTION'
        reason = build_reason("단일 신호 이상", False, flag_hr, flag_spo2,
                              flag_resp, bpm, spo2, resp)
    else:
        grade = 'NORMAL'
        reason = ""
    return grade, reason, n_vital


def follow_up(grade, reason):
    """후속 관찰(낙상 후 60초), 지속성 확인(CAUTION), 취소 유예(CRITICAL 30초)."""
    global persistence_counter, fall_followup_until, cancel_grace_until, last_critical_trigger
    now = time.time()

    # 다른 등급으로 넘어가면 관련 없는 상태는 리셋
    if grade != 'CAUTION_FALL':
        fall_followup_until = 0
    if grade != 'CAUTION':
        persistence_counter = 0
    if grade != 'CRITICAL':
        cancel_grace_until = 0
        last_critical_trigger = ""

    if grade == 'CAUTION_FALL':
        if fall_followup_until == 0:
            fall_followup_until = now + FALL_FOLLOWUP_SEC
        if imu_result['activity'] == 'active':
            fall_followup_until = 0
            return 'CAUTION', f"{reason} [복귀 감지: 활동 재개]"
        if now > fall_followup_until:
            fall_followup_until = 0
            return 'WARNING', f"{reason} [60초 무동작 → 격상]"
        return 'CAUTION_FALL', f"{reason} [{int(fall_followup_until - now)}초 남음]"

    if grade == 'CAUTION':
        persistence_counter += 1
        if persistence_counter >= PERSISTENCE_WINDOWS:
            return 'WARNING', f"{reason} [지속 확인됨]"
        return 'CAUTION', f"{reason} [{persistence_counter}/{PERSISTENCE_WINDOWS}]"

    if grade == 'CRITICAL':
        if cancel_grace_until == 0:
            cancel_grace_until = now + CANCEL_GRACE_SEC
            last_critical_trigger = reason
            return 'CRITICAL_PENDING', f"{reason} [30초 취소 유예]"
        if imu_result['activity'] == 'active':
            cancel_grace_until = 0
            return 'CAUTION', f"[자동 취소: 활동 감지] 이전: {last_critical_trigger}"
        if now > cancel_grace_until:
            cancel_grace_until = 0
            return 'CRITICAL', reason
        return 'CRITICAL_PENDING', f"{reason} [{int(cancel_grace_until - now)}초 남음]"

    return grade, reason


# ===== 화면 출력 + 보고서 트리거 =====
def evaluate_and_print():
    grade, reason, _ = evaluate_grade()
    final_grade, final_reason = follow_up(grade, reason)
    bpm, spo2, rpm = vitals['bpm'], vitals['spo2'], vitals['rpm']

    labels = {
        'NORMAL':           "NORMAL",
        'CAUTION':          f"CAUTION: {final_reason}",
        'CAUTION_FALL':     f"CAUTION (낙상 관찰): {final_reason}",
        'WARNING':          f"WARNING: {final_reason}",
        'CRITICAL_PENDING': f"CRITICAL (취소 가능): {final_reason}",
        'CRITICAL':         f"CRITICAL 발송: {final_reason}",
    }
    status = labels.get(final_grade, final_reason)

    if final_grade in ('WARNING', 'CRITICAL') and not is_ai_analyzing:
        asyncio.create_task(
            request_emergency_report(final_reason, final_grade, get_recent_logs(10)))

    bpm_s = f"{bpm:4.1f}" if bpm is not None else "  --"
    spo2_s = f"{spo2:4.1f}" if spo2 is not None else "  --"
    rpm_s = f"{rpm:4.1f}" if rpm is not None else "  --"

    eog_s = ""
    if EOG_ENABLED:
        if not eog_calib_done:
            remaining = max(0, (EOG_CALIB_SAMPLES - eog_sample_idx) // 100)
            progress = min(100, int(100 * eog_sample_idx / EOG_CALIB_SAMPLES))
            eog_s = f" | EOG 캘리브레이션 {progress}% ({remaining}s)"
        else:
            # 캘리브레이션 직후에는 모인 시간만큼 분당 횟수로 환산
            elapsed = (eog_sample_idx - EOG_CALIB_SAMPLES) / 100
            seconds_collected = max(1, min(30, elapsed))
            count = get_blink_count_30s()
            blink_per_min = int(count * (60 / seconds_collected))
            mark = "low " if count < EOG_LOW_BLINK_THRESHOLD else "norm"
            eog_s = f" | 깜빡임 {blink_per_min:2d}/min({mark})"

    print(f"\rHR:{bpm_s} | SpO2:{spo2_s} | RR:{rpm_s}{eog_s} | "
          f"{imu_result['activity']:7s} | 낙상 점수:{imu_result['fall_prob']:.2f} | "
          f"{status}     ", end="", flush=True)
    log_to_csv(bpm, spo2, rpm, imu_result['fall_prob'],
               imu_result['activity'], final_grade, final_reason)


# ===== BLE 수신 =====
def notification_handler(sender, data):
    global imu_result, eog_sample_idx
    try:
        parts = data.decode("utf-8", errors='ignore').strip().split(",")

        # 패킷 형식: ax,ay,az,gx,gy,gz,ir,red[,eog]
        if EOG_ENABLED:
            if len(parts) != 9:
                return
            eog_value = float(parts[8])
            eog_buffer.append(eog_value)
            if len(eog_buffer) > EOG_BUFFER_SIZE:
                eog_buffer.pop(0)
            eog_sample_idx += 1

            if not eog_calib_done:
                eog_calib_samples.append(eog_value)
                if eog_sample_idx >= EOG_CALIB_SAMPLES:
                    finish_eog_calibration()
            else:
                detect_blink(eog_value, eog_sample_idx)
        elif len(parts) != 8:
            return

        ax, ay, az, gx, gy, gz = (float(parts[i]) for i in range(6))
        ir, red = float(parts[6]), float(parts[7])
    except ValueError:
        return

    # IMU: 윈도우가 차면 분류 후 등급 판단
    window = imu_buffer.add(ax, ay, az, gx, gy, gz)
    if window is not None:
        imu_result = inference.predict(window)
        evaluate_and_print()

    # PPG: 손가락/피부 접촉(IR > 30000)일 때만 버퍼에 쌓음
    if ir > 30000:
        ir_buffer.append(ir)
        red_buffer.append(red)
        timestamps.append(time.time())
        if len(ir_buffer) >= BUFFER_SIZE:
            elapsed = timestamps[-1] - timestamps[0]
            rate = (len(timestamps) - 1) / elapsed if elapsed > 0 else 100.0
            run_calculate_vitals_async(ir_buffer, red_buffer, rate)
            del ir_buffer[:SLIDE_SIZE], red_buffer[:SLIDE_SIZE], timestamps[:SLIDE_SIZE]
    elif ir == 0:
        pass
    else:
        ir_buffer.clear()
        red_buffer.clear()
        timestamps.clear()
        vitals['bpm'] = vitals['spo2'] = vitals['rpm'] = None


async def run():
    print(f"'{XIAO_NAME}' 검색 중...")
    devices = await BleakScanner.discover()
    xiao = next((d for d in devices if d.name == XIAO_NAME), None)
    if not xiao:
        print("보드를 찾을 수 없습니다.")
        return
    async with BleakClient(xiao) as client:
        print("연결 성공. 스트리밍 시작")
        print("등급: NORMAL / CAUTION / WARNING / CRITICAL\n")
        await client.start_notify(DATA_UUID, notification_handler)
        while True:
            await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n시스템 종료")
