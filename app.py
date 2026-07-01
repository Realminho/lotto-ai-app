import os
import random
import time
import json
import re
from collections import Counter
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI

# =========================================================
# NVIDIA API 설정
# - Streamlit Cloud에서는 Secrets에 NVIDIA_API_KEY를 넣으세요.
# - 로컬 실행만 할 때는 아래 YOUR_API_KEY를 직접 바꿔도 됩니다.
# =========================================================
try:
    NVIDIA_API_KEY = st.secrets.get("NVIDIA_API_KEY", "YOUR_API_KEY")
except Exception:
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "YOUR_API_KEY")

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "minimaxai/minimax-m3"

OFFICIAL_API_URL = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={draw_no}"
MIRROR_ALL_URL = "https://smok95.github.io/lotto/results/all.json"
MIRROR_LATEST_URL = "https://smok95.github.io/lotto/results/latest.json"
MIRROR_DRAW_URL = "https://smok95.github.io/lotto/results/{draw_no}.json"

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.dhlottery.co.kr/",
}

st.set_page_config(
    page_title="통계 패턴 분석형 AI 로또 번호 분석/생성기 v8",
    page_icon="🎲",
    layout="wide",
)


# =========================================================
# 공통 유틸
# =========================================================
def safe_int(value, default=None):
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def format_money(value):
    value = safe_int(value, 0)
    return f"{value:,}원"


def normalize_draw(raw: Dict, source: str) -> Optional[Dict]:
    """공식 API / mirror API 데이터 형태를 앱 표준 형태로 통일"""
    if not isinstance(raw, dict):
        return None

    # 공식 API 형태
    if "drwNo" in raw or "drwtNo1" in raw:
        if raw.get("returnValue") not in (None, "success"):
            return None
        draw_no = safe_int(raw.get("drwNo"))
        numbers = [safe_int(raw.get(f"drwtNo{i}")) for i in range(1, 7)]
        bonus = safe_int(raw.get("bnusNo"))
        draw_date = raw.get("drwNoDate") or raw.get("date") or ""
        first_win = safe_int(raw.get("firstWinamnt"), 0)
        first_count = safe_int(raw.get("firstPrzwnerCo"), 0)
        total_sales = safe_int(raw.get("totSellamnt"), 0)

    # smok95 mirror 형태
    elif "draw_no" in raw or "numbers" in raw:
        draw_no = safe_int(raw.get("draw_no"))
        numbers = raw.get("numbers") or []
        numbers = [safe_int(x) for x in numbers[:6]]
        bonus = safe_int(raw.get("bonus_no") or raw.get("bonus"))
        draw_date = raw.get("date") or raw.get("draw_date") or ""
        if isinstance(draw_date, str) and "T" in draw_date:
            draw_date = draw_date.split("T")[0]

        divisions = raw.get("divisions") or []
        first_win = 0
        first_count = 0
        if isinstance(divisions, list) and len(divisions) > 0 and isinstance(divisions[0], dict):
            first_win = safe_int(divisions[0].get("prize"), 0)
            first_count = safe_int(divisions[0].get("winners"), 0)
        total_sales = safe_int(raw.get("total_sales_amount"), 0)

    else:
        return None

    if draw_no is None or len(numbers) != 6 or bonus is None:
        return None

    return {
        "회차": draw_no,
        "추첨일": str(draw_date)[:10],
        "번호1": numbers[0],
        "번호2": numbers[1],
        "번호3": numbers[2],
        "번호4": numbers[3],
        "번호5": numbers[4],
        "번호6": numbers[5],
        "보너스": bonus,
        "1등당첨금": first_win,
        "1등당첨자수": first_count,
        "총판매금액": total_sales,
        "데이터출처": source,
    }


def validate_one_draw(row: Dict) -> List[str]:
    """회차 1개 단위 기본 무결성 검사"""
    errors = []
    draw_no = safe_int(row.get("회차"))
    numbers = [safe_int(row.get(f"번호{i}")) for i in range(1, 7)]
    bonus = safe_int(row.get("보너스"))

    if draw_no is None or draw_no < 1:
        errors.append("회차 번호 오류")

    if len(numbers) != 6 or any(n is None for n in numbers):
        errors.append("당첨번호 6개 누락")
    else:
        if any(n < 1 or n > 45 for n in numbers):
            errors.append("당첨번호 범위 오류")
        if len(set(numbers)) != 6:
            errors.append("당첨번호 중복")

    if bonus is None or bonus < 1 or bonus > 45:
        errors.append("보너스 번호 범위 오류")
    elif bonus in numbers:
        errors.append("보너스 번호가 당첨번호와 중복")

    return errors


def validate_history_df(df: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
    """전체 데이터 기본 무결성 검사"""
    issues = []

    if df.empty:
        return False, pd.DataFrame([{"구분": "전체", "내용": "데이터가 비어 있습니다."}])

    needed = ["회차", "번호1", "번호2", "번호3", "번호4", "번호5", "번호6", "보너스"]
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        issues.append({"구분": "컬럼", "내용": f"필수 컬럼 누락: {missing_cols}"})
        return False, pd.DataFrame(issues)

    for _, row in df.iterrows():
        row_errors = validate_one_draw(row.to_dict())
        for err in row_errors:
            issues.append({"구분": f"{int(row['회차'])}회", "내용": err})

    draw_nos = sorted(df["회차"].astype(int).tolist())
    expected = list(range(min(draw_nos), max(draw_nos) + 1))
    missing_draws = sorted(set(expected) - set(draw_nos))
    duplicated_draws = df[df.duplicated(subset=["회차"], keep=False)]["회차"].astype(int).unique().tolist()

    if missing_draws:
        sample = missing_draws[:20]
        more = "..." if len(missing_draws) > 20 else ""
        issues.append({"구분": "회차 연속성", "내용": f"누락 회차: {sample}{more}"})

    if duplicated_draws:
        issues.append({"구분": "회차 중복", "내용": f"중복 회차: {sorted(duplicated_draws)}"})

    if len(issues) == 0:
        return True, pd.DataFrame([{"구분": "무결성 검사", "내용": "통과"}])
    return False, pd.DataFrame(issues)


# =========================================================
# 데이터 불러오기
# =========================================================
def request_json(url: str, timeout: int = 15) -> Tuple[Optional[object], Optional[str]]:
    try:
        r = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
        text = r.text.strip()
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        if text.startswith("<"):
            return None, "JSON이 아니라 HTML이 반환되었습니다. 접속 대기/차단 페이지일 수 있습니다."
        return r.json(), None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=60 * 60)
def fetch_official_draw(draw_no: int) -> Tuple[Optional[Dict], Optional[str]]:
    raw, err = request_json(OFFICIAL_API_URL.format(draw_no=draw_no), timeout=12)
    if err:
        return None, err
    normalized = normalize_draw(raw, "동행복권 공식")
    if normalized is None:
        return None, "공식 API 응답을 해석하지 못했습니다."
    return normalized, None


@st.cache_data(ttl=60 * 60)
def fetch_mirror_latest() -> Tuple[Optional[Dict], Optional[str]]:
    raw, err = request_json(MIRROR_LATEST_URL, timeout=15)
    if err:
        return None, err
    normalized = normalize_draw(raw, "GitHub mirror")
    if normalized is None:
        return None, "mirror latest 응답을 해석하지 못했습니다."
    return normalized, None


@st.cache_data(ttl=60 * 60)
def fetch_mirror_all() -> Tuple[pd.DataFrame, Optional[str]]:
    raw, err = request_json(MIRROR_ALL_URL, timeout=30)
    if err:
        return pd.DataFrame(), err

    rows = []
    if isinstance(raw, list):
        source_items = raw
    elif isinstance(raw, dict):
        # 혹시 dict 형태로 감싸져 있을 때 대비
        source_items = raw.get("data") or raw.get("results") or raw.get("draws") or list(raw.values())
    else:
        source_items = []

    for item in source_items:
        normalized = normalize_draw(item, "GitHub mirror")
        if normalized:
            rows.append(normalized)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, "mirror all 데이터를 해석하지 못했습니다."

    df = df.drop_duplicates(subset=["회차"], keep="last")
    df = df.sort_values("회차", ascending=True).reset_index(drop=True)
    return df, None


@st.cache_data(ttl=60 * 60)
def fetch_mirror_by_draws(latest_draw: int) -> Tuple[pd.DataFrame, Optional[str]]:
    rows = []
    errors = []
    for draw_no in range(1, latest_draw + 1):
        raw, err = request_json(MIRROR_DRAW_URL.format(draw_no=draw_no), timeout=10)
        if err:
            errors.append(f"{draw_no}회: {err}")
            continue
        normalized = normalize_draw(raw, "GitHub mirror")
        if normalized:
            rows.append(normalized)
        time.sleep(0.02)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, "; ".join(errors[:5]) if errors else "회차별 mirror 데이터를 해석하지 못했습니다."
    df = df.drop_duplicates(subset=["회차"], keep="last")
    df = df.sort_values("회차", ascending=True).reset_index(drop=True)
    return df, None


@st.cache_data(ttl=60 * 60)
def fetch_official_history_slow(latest_draw: int) -> Tuple[pd.DataFrame, Optional[str]]:
    rows = []
    errors = []
    progress_every = 50
    for draw_no in range(1, latest_draw + 1):
        row, err = fetch_official_draw(draw_no)
        if row:
            rows.append(row)
        else:
            errors.append(f"{draw_no}회: {err}")
        if draw_no % progress_every == 0:
            time.sleep(0.2)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, "; ".join(errors[:5]) if errors else "공식 데이터를 불러오지 못했습니다."
    df = df.drop_duplicates(subset=["회차"], keep="last")
    df = df.sort_values("회차", ascending=True).reset_index(drop=True)
    return df, None


def compare_draw_rows(mirror_row: Dict, official_row: Dict) -> Tuple[bool, Dict]:
    fields = ["회차", "추첨일", "번호1", "번호2", "번호3", "번호4", "번호5", "번호6", "보너스"]
    details = {"회차": mirror_row.get("회차")}
    ok = True
    for f in fields:
        m = mirror_row.get(f)
        o = official_row.get(f)
        if str(m) != str(o):
            ok = False
        details[f"mirror_{f}"] = m
        details[f"official_{f}"] = o
    details["일치여부"] = "일치" if ok else "불일치"
    return ok, details


def verify_with_official(history_df: pd.DataFrame, count: int) -> Tuple[str, pd.DataFrame, List[int], Optional[str]]:
    """최신 N개 회차를 공식 API와 비교"""
    if history_df.empty:
        return "검증 불가", pd.DataFrame(), [], "비교할 데이터가 없습니다."

    latest_draw = int(history_df["회차"].max())
    start_draw = max(1, latest_draw - count + 1)
    target_draws = list(range(start_draw, latest_draw + 1))

    comparison_rows = []
    official_success = []
    official_errors = []
    all_ok = True

    for draw_no in target_draws:
        mirror_rows = history_df[history_df["회차"].astype(int) == draw_no]
        if mirror_rows.empty:
            all_ok = False
            comparison_rows.append({"회차": draw_no, "일치여부": "mirror 데이터 없음"})
            continue

        official_row, err = fetch_official_draw(draw_no)
        if official_row is None:
            all_ok = False
            official_errors.append(f"{draw_no}회: {err}")
            comparison_rows.append({"회차": draw_no, "일치여부": "공식 조회 실패", "오류": err})
            continue

        official_success.append(draw_no)
        ok, detail = compare_draw_rows(mirror_rows.iloc[0].to_dict(), official_row)
        if not ok:
            all_ok = False
        comparison_rows.append(detail)
        time.sleep(0.05)

    compare_df = pd.DataFrame(comparison_rows)

    if len(official_success) == 0:
        return "검증 불가", compare_df, official_success, "; ".join(official_errors[:5])

    if all_ok and len(official_success) == len(target_draws):
        return "공식 검증 통과", compare_df, official_success, None

    if any(str(x.get("일치여부")) == "불일치" for x in comparison_rows):
        return "공식값과 불일치 발견", compare_df, official_success, "; ".join(official_errors[:5]) if official_errors else None

    return "일부만 검증됨", compare_df, official_success, "; ".join(official_errors[:5]) if official_errors else None


# =========================================================
# 통계 및 번호 생성
# =========================================================
def make_count_table(history_df: pd.DataFrame) -> pd.DataFrame:
    main_counter = Counter()
    bonus_counter = Counter()

    for _, row in history_df.iterrows():
        main_numbers = [safe_int(row[f"번호{i}"]) for i in range(1, 7)]
        for number in main_numbers:
            if number is not None:
                main_counter[number] += 1
        bonus = safe_int(row.get("보너스"))
        if bonus is not None:
            bonus_counter[bonus] += 1

    rows = []
    for number in range(1, 46):
        main_count = main_counter[number]
        bonus_count = bonus_counter[number]
        rows.append({
            "번호": number,
            "당첨횟수": main_count,
            "보너스횟수": bonus_count,
            "보너스포함횟수": main_count + bonus_count,
        })

    return pd.DataFrame(rows).sort_values("당첨횟수", ascending=False).reset_index(drop=True)


def weighted_sample_without_replacement(numbers, weights, k=6):
    numbers = list(numbers)
    weights = [max(1, int(w)) for w in weights]
    selected = []

    for _ in range(k):
        if len(numbers) == 0:
            break
        chosen = random.choices(numbers, weights=weights, k=1)[0]
        idx = numbers.index(chosen)
        selected.append(chosen)
        numbers.pop(idx)
        weights.pop(idx)
    return sorted(selected)


def generate_lotto_numbers(count_df, mode, min_count, top_n, use_weight, set_count):
    if mode == "전체 1~45에서 생성":
        pool_df = count_df.copy()
    elif mode == "당첨횟수 N회 이상 번호에서만 생성":
        pool_df = count_df[count_df["당첨횟수"] >= min_count].copy()
    elif mode == "당첨횟수 상위 N개 번호에서만 생성":
        pool_df = count_df.sort_values("당첨횟수", ascending=False).head(top_n).copy()
    else:
        pool_df = count_df.copy()

    pool_numbers = pool_df["번호"].astype(int).tolist()
    if len(pool_numbers) < 6:
        return [], pool_df

    results = []
    for _ in range(set_count):
        if use_weight:
            nums = weighted_sample_without_replacement(pool_numbers, pool_df["당첨횟수"].tolist(), 6)
        else:
            nums = sorted(random.sample(pool_numbers, 6))
        results.append(nums)
    return results, pool_df


def minmax_0_100(series: pd.Series) -> pd.Series:
    """값을 0~100 점수로 변환"""
    s = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
    min_v = float(s.min())
    max_v = float(s.max())
    if max_v == min_v:
        return pd.Series([50.0] * len(s), index=s.index)
    return ((s - min_v) / (max_v - min_v) * 100).round(2)


def extract_main_numbers(row: Dict) -> List[int]:
    nums = []
    for i in range(1, 7):
        n = safe_int(row.get(f"번호{i}"))
        if n is not None:
            nums.append(int(n))
    return nums


def count_consecutive_pairs(numbers: List[int]) -> int:
    nums = sorted(numbers)
    return sum(1 for a, b in zip(nums, nums[1:]) if b - a == 1)


def max_same_last_digit(numbers: List[int]) -> int:
    counter = Counter([n % 10 for n in numbers])
    return max(counter.values()) if counter else 0


PRIME_NUMBERS = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}


def max_consecutive_run(numbers: List[int]) -> int:
    nums = sorted(numbers)
    if not nums:
        return 0
    best = 1
    current = 1
    for a, b in zip(nums, nums[1:]):
        if b - a == 1:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def count_by_ranges(numbers: List[int], ranges: List[Tuple[int, int]]) -> List[int]:
    nums = [int(n) for n in numbers]
    return [sum(1 for n in nums if start <= n <= end) for start, end in ranges]


def dist_label(counts: List[int]) -> str:
    return "-".join(str(int(x)) for x in counts)


def number_range_label(n: int) -> str:
    n = int(n)
    if 1 <= n <= 9:
        return "1~9"
    if 10 <= n <= 19:
        return "10~19"
    if 20 <= n <= 29:
        return "20~29"
    if 30 <= n <= 39:
        return "30~39"
    return "40~45"


def bucket_label(value: float, size: int) -> str:
    start = int(value // size) * size
    return f"{start}~{start + size - 1}"


def gap_label(gap: float) -> str:
    gap = float(gap)
    if gap <= 3:
        return "1~3"
    if gap <= 6:
        return "4~6"
    if gap <= 10:
        return "7~10"
    if gap <= 15:
        return "11~15"
    return "16이상"


def gap_pattern_label(gaps: List[int]) -> str:
    labels = []
    for g in gaps:
        if g <= 3:
            labels.append("초근접")
        elif g <= 6:
            labels.append("근접")
        elif g <= 10:
            labels.append("보통")
        elif g <= 15:
            labels.append("넓음")
        else:
            labels.append("매우넓음")
    return "-".join(labels)


def combo_pattern_values_from_features(features: Dict) -> Dict[str, str]:
    """조합 하나를 여러 종류의 패턴값으로 변환"""
    return {
        "홀짝 개수": f"홀수 {features['홀수개수']}개 / 짝수 {features['짝수개수']}개",
        "홀짝 순서(오름차순)": features["홀짝순서"],
        "저번호/고번호 개수": f"저번호 {features['저번호개수']}개 / 고번호 {features['고번호개수']}개",
        "저고 순서(오름차순)": features["저고순서"],
        "5구간 분포(1~9/10~19/20~29/30~39/40~45)": features["5구간분포"],
        "3구간 분포(1~15/16~30/31~45)": features["3구간분포"],
        "합계 10단위 구간": features["합계10구간"],
        "합계 20단위 구간": features["합계20구간"],
        "연속번호쌍 개수": f"{features['연속번호쌍']}쌍",
        "최대 연속 길이": f"{features['최대연속길이']}개 연속",
        "끝자리 최대 중복": f"최대 {features['같은끝자리최대']}개",
        "끝자리 종류 수": f"{features['끝자리종류수']}종류",
        "소수 개수": f"{features['소수개수']}개",
        "3의 배수 개수": f"{features['3배수개수']}개",
        "5의 배수 개수": f"{features['5배수개수']}개",
        "평균 간격 구간": features["평균간격구간"],
        "최대 간격 구간": features["최대간격구간"],
        "간격 크기 패턴": features["간격패턴"],
        "최소번호 구간": features["최소번호구간"],
        "최대번호 구간": features["최대번호구간"],
    }


def summarize_combo_patterns(features: Dict) -> str:
    return (
        f"홀{features['홀수개수']}/짝{features['짝수개수']}, "
        f"저{features['저번호개수']}/고{features['고번호개수']}, "
        f"5구간 {features['5구간분포']}, "
        f"합계 {features['합계']}, "
        f"소수 {features['소수개수']}개, "
        f"3배수 {features['3배수개수']}개, "
        f"연속 {features['연속번호쌍']}쌍, "
        f"끝자리최대 {features['같은끝자리최대']}개"
    )


def make_pattern_analysis(history_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """과거 당첨 조합에서 다양한 패턴을 전부 뽑아 빈도순으로 정리"""
    rows = []
    prev_nums = None
    for _, row in history_df.sort_values("회차", ascending=True).iterrows():
        nums = extract_main_numbers(row.to_dict())
        if len(nums) != 6:
            continue
        features = combo_basic_features(nums)
        pattern_values = combo_pattern_values_from_features(features)
        if prev_nums is not None:
            overlap = len(set(nums) & set(prev_nums))
            pattern_values["전회차 번호 재등장 개수"] = f"{overlap}개"
        for ptype, pvalue in pattern_values.items():
            rows.append({
                "회차": int(row["회차"]),
                "당첨번호": "-".join(map(str, sorted(nums))),
                "패턴종류": ptype,
                "패턴값": pvalue,
            })
        prev_nums = nums

    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = (
        raw_df.groupby(["패턴종류", "패턴값"])
        .size()
        .reset_index(name="등장횟수")
    )
    totals = raw_df.groupby("패턴종류").size().to_dict()
    summary["비율(%)"] = summary.apply(
        lambda r: round(float(r["등장횟수"]) / max(1, totals.get(r["패턴종류"], 1)) * 100, 2),
        axis=1,
    )
    summary = summary.sort_values(["패턴종류", "등장횟수", "패턴값"], ascending=[True, False, True]).reset_index(drop=True)
    summary["패턴순위"] = summary.groupby("패턴종류")["등장횟수"].rank(method="first", ascending=False).astype(int)
    summary = summary[["패턴종류", "패턴순위", "패턴값", "등장횟수", "비율(%)"]]
    return summary, raw_df


def make_pattern_maps(pattern_summary_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    maps = {}
    if pattern_summary_df is None or pattern_summary_df.empty:
        return maps
    for ptype, group in pattern_summary_df.groupby("패턴종류"):
        maps[ptype] = dict(zip(group["패턴값"].astype(str), group["등장횟수"].astype(int)))
    return maps


def score_pattern_value(pattern_maps: Dict[str, Dict[str, int]], ptype: str, pvalue: str) -> float:
    value_map = pattern_maps.get(ptype, {})
    if not value_map:
        return 70.0
    max_count = max(value_map.values()) if value_map else 1
    count = int(value_map.get(str(pvalue), 0))
    if count <= 0:
        return 52.0
    # 가장 많이 나온 패턴은 100점, 드문 패턴도 완전히 0점 처리하지 않고 55점 이상 부여
    return round(55.0 + 45.0 * (count / max(1, max_count)), 2)


def calculate_pattern_score(features: Dict, profile: Dict) -> Tuple[float, Dict[str, float]]:
    pattern_maps = profile.get("pattern_maps", {}) if isinstance(profile, dict) else {}
    pattern_values = combo_pattern_values_from_features(features)
    # 너무 세부적인 순서 패턴은 참고만 하고, 점수에는 대표 패턴 위주로 반영
    scoring_types = [
        "홀짝 개수",
        "저번호/고번호 개수",
        "5구간 분포(1~9/10~19/20~29/30~39/40~45)",
        "3구간 분포(1~15/16~30/31~45)",
        "합계 20단위 구간",
        "연속번호쌍 개수",
        "끝자리 최대 중복",
        "끝자리 종류 수",
        "소수 개수",
        "3의 배수 개수",
        "5의 배수 개수",
        "평균 간격 구간",
        "최대 간격 구간",
        "최소번호 구간",
        "최대번호 구간",
    ]
    scores = {}
    for ptype in scoring_types:
        if ptype in pattern_values:
            scores[ptype] = score_pattern_value(pattern_maps, ptype, pattern_values[ptype])
    if not scores:
        return 70.0, {}
    return round(sum(scores.values()) / len(scores), 2), scores


def make_pattern_prompt_text(pattern_summary_df: pd.DataFrame, top_n_per_type: int = 5) -> str:
    if pattern_summary_df is None or pattern_summary_df.empty:
        return "패턴 빈도표 없음"
    lines = []
    for ptype, group in pattern_summary_df.sort_values(["패턴종류", "패턴순위"]).groupby("패턴종류"):
        top = group.head(int(top_n_per_type))
        values = [f"{r['패턴값']}({int(r['등장횟수'])}회, {r['비율(%)']}%)" for _, r in top.iterrows()]
        lines.append(f"- {ptype}: " + " / ".join(values))
    return "\n".join(lines)


def combo_basic_features(numbers: List[int]) -> Dict:
    nums = sorted([int(n) for n in numbers])
    odd_count = sum(1 for n in nums if n % 2 == 1)
    low_count = sum(1 for n in nums if n <= 22)
    gaps = [b - a for a, b in zip(nums, nums[1:])]
    avg_gap = sum(gaps) / len(gaps) if gaps else 0
    max_gap = max(gaps) if gaps else 0
    range5_counts = count_by_ranges(nums, [(1, 9), (10, 19), (20, 29), (30, 39), (40, 45)])
    range3_counts = count_by_ranges(nums, [(1, 15), (16, 30), (31, 45)])
    total_sum = sum(nums)
    return {
        "합계": total_sum,
        "홀수개수": odd_count,
        "짝수개수": 6 - odd_count,
        "홀짝순서": "-".join("홀" if n % 2 == 1 else "짝" for n in nums),
        "저번호개수": low_count,
        "고번호개수": 6 - low_count,
        "저고순서": "-".join("저" if n <= 22 else "고" for n in nums),
        "5구간분포": dist_label(range5_counts),
        "3구간분포": dist_label(range3_counts),
        "합계10구간": bucket_label(total_sum, 10),
        "합계20구간": bucket_label(total_sum, 20),
        "연속번호쌍": count_consecutive_pairs(nums),
        "최대연속길이": max_consecutive_run(nums),
        "같은끝자리최대": max_same_last_digit(nums),
        "끝자리종류수": len(set(n % 10 for n in nums)),
        "소수개수": sum(1 for n in nums if n in PRIME_NUMBERS),
        "3배수개수": sum(1 for n in nums if n % 3 == 0),
        "5배수개수": sum(1 for n in nums if n % 5 == 0),
        "평균간격": round(avg_gap, 2),
        "평균간격구간": gap_label(avg_gap),
        "최대간격": int(max_gap),
        "최대간격구간": gap_label(max_gap),
        "간격패턴": gap_pattern_label(gaps),
        "최소번호구간": number_range_label(min(nums)),
        "최대번호구간": number_range_label(max(nums)),
    }


def make_historical_combo_profile(history_df: pd.DataFrame) -> Dict:
    rows = []
    for _, row in history_df.iterrows():
        nums = extract_main_numbers(row.to_dict())
        if len(nums) == 6:
            rows.append(combo_basic_features(nums))
    df = pd.DataFrame(rows)
    pattern_summary_df, _ = make_pattern_analysis(history_df)
    pattern_maps = make_pattern_maps(pattern_summary_df)
    pattern_top_text = make_pattern_prompt_text(pattern_summary_df, top_n_per_type=5)

    if df.empty:
        return {
            "sum_q10": 90, "sum_q25": 105, "sum_q75": 170, "sum_q90": 190,
            "common_odd_counts": [2, 3, 4],
            "common_low_counts": [2, 3, 4],
            "pattern_summary_df": pattern_summary_df,
            "pattern_maps": pattern_maps,
            "pattern_top_text": pattern_top_text,
        }
    odd_common = df["홀수개수"].value_counts().index.astype(int).tolist()
    low_common = df["저번호개수"].value_counts().index.astype(int).tolist()
    return {
        "sum_q10": float(df["합계"].quantile(0.10)),
        "sum_q25": float(df["합계"].quantile(0.25)),
        "sum_q75": float(df["합계"].quantile(0.75)),
        "sum_q90": float(df["합계"].quantile(0.90)),
        # 이제 상위 3개만이 아니라 실제 등장한 모든 개수 패턴을 빈도순으로 보관
        "common_odd_counts": odd_common or [2, 3, 4],
        "common_low_counts": low_common or [2, 3, 4],
        "pattern_summary_df": pattern_summary_df,
        "pattern_maps": pattern_maps,
        "pattern_top_text": pattern_top_text,
    }


def make_statistical_score_table(
    history_df: pd.DataFrame,
    count_df: pd.DataFrame,
    recent_window: int,
    weight_total: int,
    weight_recent: int,
    weight_gap: int,
    weight_bonus: int,
) -> pd.DataFrame:
    """장기 빈도, 최근 흐름, 미출현 간격, 보너스 데이터를 합쳐 1~45 통계점수 생성"""
    total_draws = max(1, len(history_df))
    latest_draw = int(history_df["회차"].max())
    recent_window = max(1, min(int(recent_window), total_draws))
    recent_df = history_df.sort_values("회차", ascending=True).tail(recent_window)

    recent_counter = Counter()
    last_seen = {n: None for n in range(1, 46)}

    for _, row in history_df.iterrows():
        draw_no = int(row["회차"])
        nums = extract_main_numbers(row.to_dict())
        for n in nums:
            last_seen[n] = draw_no

    for _, row in recent_df.iterrows():
        nums = extract_main_numbers(row.to_dict())
        for n in nums:
            recent_counter[n] += 1

    base = count_df.copy().sort_values("번호").reset_index(drop=True)
    rows = []
    for _, row in base.iterrows():
        n = int(row["번호"])
        total_count = int(row["당첨횟수"])
        bonus_count = int(row["보너스횟수"])
        recent_count = int(recent_counter[n])
        if last_seen[n] is None:
            gap = latest_draw
        else:
            gap = latest_draw - int(last_seen[n])
        avg_gap = total_draws / max(1, total_count)
        gap_ratio = min(3.0, gap / max(1.0, avg_gap))
        expected_recent = recent_window * 6 / 45
        recent_over_expected = recent_count - expected_recent
        rows.append({
            "번호": n,
            "전체당첨횟수": total_count,
            "전체출현률": round(total_count / total_draws * 100, 2),
            f"최근{recent_window}회출현횟수": recent_count,
            f"최근{recent_window}회기대대비": round(recent_over_expected, 2),
            "보너스횟수": bonus_count,
            "마지막출현후경과회차": int(gap),
            "평균출현간격": round(avg_gap, 2),
            "미출현간격비율": round(gap_ratio, 2),
        })

    stats_df = pd.DataFrame(rows)
    stats_df["장기빈도점수"] = minmax_0_100(stats_df["전체당첨횟수"])
    stats_df["최근흐름점수"] = minmax_0_100(stats_df[f"최근{recent_window}회출현횟수"])
    stats_df["미출현간격점수"] = minmax_0_100(stats_df["미출현간격비율"])
    stats_df["보너스점수"] = minmax_0_100(stats_df["보너스횟수"])

    total_weight = max(1, int(weight_total) + int(weight_recent) + int(weight_gap) + int(weight_bonus))
    stats_df["통계점수"] = (
        stats_df["장기빈도점수"] * int(weight_total)
        + stats_df["최근흐름점수"] * int(weight_recent)
        + stats_df["미출현간격점수"] * int(weight_gap)
        + stats_df["보너스점수"] * int(weight_bonus)
    ) / total_weight
    stats_df["통계점수"] = stats_df["통계점수"].round(2)
    return stats_df.sort_values("통계점수", ascending=False).reset_index(drop=True)


def evaluate_statistical_combo(numbers: List[int], score_map: Dict[int, float], profile: Dict) -> Tuple[float, Dict]:
    nums = sorted([int(n) for n in numbers])
    features = combo_basic_features(nums)
    number_score = sum(float(score_map.get(n, 0)) for n in nums) / 6

    total_sum = features["합계"]
    if profile["sum_q25"] <= total_sum <= profile["sum_q75"]:
        sum_score = 100
    elif profile["sum_q10"] <= total_sum <= profile["sum_q90"]:
        sum_score = 82
    else:
        sum_score = 58

    pattern_score, pattern_detail_scores = calculate_pattern_score(features, profile)
    odd_score = pattern_detail_scores.get("홀짝 개수", 72)
    low_score = pattern_detail_scores.get("저번호/고번호 개수", 72)

    consecutive_pairs = features["연속번호쌍"]
    if consecutive_pairs <= 1:
        consecutive_score = 100
    elif consecutive_pairs == 2:
        consecutive_score = 80
    else:
        consecutive_score = 55

    if features["같은끝자리최대"] <= 2:
        end_digit_score = 100
    elif features["같은끝자리최대"] == 3:
        end_digit_score = 78
    else:
        end_digit_score = 55

    # v8: 번호 자체의 통계점수 + 합계 + 다양한 조합 패턴 빈도점수를 함께 반영
    combo_score = (
        number_score * 0.60
        + sum_score * 0.10
        + pattern_score * 0.22
        + consecutive_score * 0.04
        + end_digit_score * 0.04
    )
    details = {
        **features,
        "번호평균점수": round(number_score, 2),
        "합계균형점수": round(sum_score, 2),
        "패턴종합점수": round(pattern_score, 2),
        "패턴상세점수": pattern_detail_scores,
        "패턴요약": summarize_combo_patterns(features),
        "홀짝균형점수": round(odd_score, 2),
        "저고균형점수": round(low_score, 2),
        "연속번호점수": round(consecutive_score, 2),
        "끝자리분산점수": round(end_digit_score, 2),
    }
    return round(combo_score, 2), details


def build_combo_reason(numbers: List[int], stats_df: pd.DataFrame, recent_window: int) -> str:
    nums = sorted([int(n) for n in numbers])
    score_lookup = stats_df.set_index("번호")["통계점수"].to_dict()
    recent_col = f"최근{recent_window}회출현횟수"
    top_nums = sorted(nums, key=lambda n: score_lookup.get(n, 0), reverse=True)[:3]
    avg_score = sum(score_lookup.get(n, 0) for n in nums) / 6
    recent_hits = int(stats_df[stats_df["번호"].isin(nums)][recent_col].sum()) if recent_col in stats_df.columns else 0
    features = combo_basic_features(nums)
    return (
        f"통계점수 상위 번호 {top_nums} 중심, 조합 평균점수 {avg_score:.1f}, "
        f"최근 {recent_window}회 내 출현 누적 {recent_hits}회, "
        f"패턴: {summarize_combo_patterns(features)}"
    )


def generate_statistical_recommendations(
    history_df: pd.DataFrame,
    count_df: pd.DataFrame,
    recent_window: int,
    weight_total: int,
    weight_recent: int,
    weight_gap: int,
    weight_bonus: int,
    candidate_count: int,
    recommend_count: int,
    top_pool_size: int,
    include_top_six: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    stats_df = make_statistical_score_table(
        history_df=history_df,
        count_df=count_df,
        recent_window=recent_window,
        weight_total=weight_total,
        weight_recent=weight_recent,
        weight_gap=weight_gap,
        weight_bonus=weight_bonus,
    )
    profile = make_historical_combo_profile(history_df)
    pool_df = stats_df.head(int(top_pool_size)).copy()
    numbers = pool_df["번호"].astype(int).tolist()
    weights = [max(1.0, float(x)) ** 1.25 for x in pool_df["통계점수"].tolist()]
    score_map = stats_df.set_index("번호")["통계점수"].to_dict()

    candidates = {}
    if include_top_six:
        top_six = tuple(sorted(stats_df.head(6)["번호"].astype(int).tolist()))
        candidates[top_six] = True

    candidate_count = int(candidate_count)
    for _ in range(candidate_count):
        nums = tuple(weighted_sample_without_replacement(numbers, weights, 6))
        candidates[nums] = True

    rows = []
    for combo in candidates.keys():
        score, details = evaluate_statistical_combo(list(combo), score_map, profile)
        nums = list(combo)
        rows.append({
            "추천순위": 0,
            "번호1": nums[0],
            "번호2": nums[1],
            "번호3": nums[2],
            "번호4": nums[3],
            "번호5": nums[4],
            "번호6": nums[5],
            "조합통계점수": score,
            "합계": details["합계"],
            "홀짝": f"{details['홀수개수']}:{details['짝수개수']}",
            "저고": f"{details['저번호개수']}:{details['고번호개수']}",
            "구간분포5": details["5구간분포"],
            "소수개수": details["소수개수"],
            "3배수개수": details["3배수개수"],
            "5배수개수": details["5배수개수"],
            "연속번호쌍": details["연속번호쌍"],
            "끝자리최대중복": details["같은끝자리최대"],
            "패턴종합점수": details["패턴종합점수"],
            "패턴요약": details["패턴요약"],
            "추천이유": build_combo_reason(nums, stats_df, recent_window),
        })

    reco_df = pd.DataFrame(rows).sort_values("조합통계점수", ascending=False).head(int(recommend_count)).reset_index(drop=True)
    if not reco_df.empty:
        reco_df["추천순위"] = range(1, len(reco_df) + 1)
    return reco_df, stats_df, profile



def combo_numbers_from_reco_row(row: Dict) -> List[int]:
    nums = []
    for i in range(1, 7):
        n = safe_int(row.get(f"번호{i}"))
        if n is not None:
            nums.append(int(n))
    return sorted(nums)


def make_combo_numeric_detail_df(reco_df: pd.DataFrame, stats_df: pd.DataFrame, recent_window: int) -> pd.DataFrame:
    """추천 조합별 수치 설명용 요약표"""
    if reco_df is None or reco_df.empty or stats_df is None or stats_df.empty:
        return pd.DataFrame()

    recent_col = f"최근{recent_window}회출현횟수"
    stat_index = stats_df.set_index("번호")
    rows = []
    for _, row in reco_df.iterrows():
        nums = combo_numbers_from_reco_row(row.to_dict())
        selected = stat_index.loc[[n for n in nums if n in stat_index.index]].copy()
        if selected.empty:
            continue
        rows.append({
            "추천순위": safe_int(row.get("추천순위"), 0),
            "조합": "-".join(map(str, nums)),
            "조합통계점수": row.get("조합통계점수", 0),
            "번호평균통계점수": round(float(selected["통계점수"].mean()), 2),
            "장기빈도점수평균": round(float(selected["장기빈도점수"].mean()), 2),
            "최근흐름점수평균": round(float(selected["최근흐름점수"].mean()), 2),
            "미출현간격점수평균": round(float(selected["미출현간격점수"].mean()), 2),
            "보너스점수평균": round(float(selected["보너스점수"].mean()), 2),
            "전체당첨횟수합": int(selected["전체당첨횟수"].sum()) if "전체당첨횟수" in selected.columns else 0,
            f"최근{recent_window}회출현합": int(selected[recent_col].sum()) if recent_col in selected.columns else 0,
            "평균미출현간격비율": round(float(selected["미출현간격비율"].mean()), 2) if "미출현간격비율" in selected.columns else 0,
            "합계": row.get("합계", 0),
            "홀짝": row.get("홀짝", ""),
            "저고": row.get("저고", ""),
            "구간분포5": row.get("구간분포5", ""),
            "소수개수": row.get("소수개수", ""),
            "3배수개수": row.get("3배수개수", ""),
            "5배수개수": row.get("5배수개수", ""),
            "패턴종합점수": row.get("패턴종합점수", ""),
            "패턴요약": row.get("패턴요약", ""),
            "연속번호쌍": row.get("연속번호쌍", 0),
            "끝자리최대중복": row.get("끝자리최대중복", 0),
        })
    return pd.DataFrame(rows)


def make_number_numeric_detail_df(reco_df: pd.DataFrame, stats_df: pd.DataFrame, recent_window: int) -> pd.DataFrame:
    """추천 조합에 들어간 개별 번호의 수치 근거표"""
    if reco_df is None or reco_df.empty or stats_df is None or stats_df.empty:
        return pd.DataFrame()

    recent_col = f"최근{recent_window}회출현횟수"
    keep_cols = [
        "번호", "통계점수", "전체당첨횟수", "전체출현률", recent_col,
        f"최근{recent_window}회기대대비", "마지막출현후경과회차", "평균출현간격", "미출현간격비율",
        "장기빈도점수", "최근흐름점수", "미출현간격점수", "보너스점수",
    ]
    available_cols = [c for c in keep_cols if c in stats_df.columns]
    stat_index = stats_df.set_index("번호")
    rows = []
    for _, row in reco_df.iterrows():
        rank = safe_int(row.get("추천순위"), 0)
        combo = "-".join(map(str, combo_numbers_from_reco_row(row.to_dict())))
        for n in combo_numbers_from_reco_row(row.to_dict()):
            if n not in stat_index.index:
                continue
            d = stat_index.loc[n].to_dict()
            one = {"추천순위": rank, "조합": combo, "번호": n}
            for c in available_cols:
                if c != "번호":
                    one[c] = d.get(c)
            rows.append(one)
    return pd.DataFrame(rows)


def check_nvidia_api_status(api_key: str) -> Tuple[bool, str]:
    """NVIDIA API 키가 정상인지 짧게 확인"""
    if not api_key or api_key == "YOUR_API_KEY":
        return False, "API 키가 입력되지 않았습니다. Streamlit Secrets 또는 왼쪽 입력칸에 NVIDIA_API_KEY를 넣어주세요."
    try:
        client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)
        response = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[{"role": "user", "content": "정상 연결이면 OK라고만 답해."}],
            temperature=0,
            max_tokens=10,
        )
        text = response.choices[0].message.content or ""
        return True, f"정상 연결됨: {text.strip()}"
    except Exception as e:
        msg = str(e)
        lower = msg.lower()
        if "401" in msg or "unauthorized" in lower or "invalid" in lower:
            return False, "API 키 인증 실패로 보입니다. NVIDIA에서 새 API 키를 발급한 뒤 Streamlit Secrets에 직접 저장해야 합니다. 앱이 자동으로 재발급하거나 Secrets를 수정할 수는 없습니다.\n\n오류: " + msg
        if "403" in msg or "forbidden" in lower:
            return False, "권한 또는 모델 접근 제한 오류로 보입니다. NVIDIA 계정/모델 접근 권한을 확인하세요.\n\n오류: " + msg
        if "429" in msg or "rate" in lower or "quota" in lower:
            return False, "사용량 제한 또는 호출 제한으로 보입니다. 잠시 후 다시 시도하거나 NVIDIA 계정의 사용량 한도를 확인하세요.\n\n오류: " + msg
        return False, "NVIDIA API 호출 중 오류가 발생했습니다.\n\n오류: " + msg


def ask_nvidia_stat_ai(api_key, reco_df, stats_df, latest_draw, verification_status, recent_window, combo_profile=None):
    if not api_key or api_key == "YOUR_API_KEY":
        return "NVIDIA API 키가 설정되지 않았습니다. Streamlit Secrets에 NVIDIA_API_KEY를 넣거나 왼쪽 입력창에 키를 입력해주세요."

    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)
    reco_text = reco_df.head(8).to_string(index=False)
    top_text = stats_df.head(20).to_string(index=False)
    combo_detail_df = make_combo_numeric_detail_df(reco_df.head(8), stats_df, recent_window)
    number_detail_df = make_number_numeric_detail_df(reco_df.head(5), stats_df, recent_window)
    combo_detail_text = combo_detail_df.to_string(index=False) if not combo_detail_df.empty else "없음"
    number_detail_text = number_detail_df.to_string(index=False) if not number_detail_df.empty else "없음"
    profile = combo_profile or {}
    pattern_top_text = profile.get("pattern_top_text", "패턴 빈도표 없음")

    prompt = f"""
너는 로또 번호를 '통계적으로 설명만 하는' 분석 보조 AI야.
아래 수치표를 근거로 한국어로 자세히 설명해줘.
절대 당첨 보장, 당첨 확률 상승 확정, 예측 성공 같은 표현은 쓰면 안 돼.

[기본 정보]
- 최신 반영 회차: {latest_draw}회
- 데이터 검증 상태: {verification_status}
- 최근 흐름 분석 범위: 최근 {recent_window}회
- 과거 조합 합계 IQR: Q25={profile.get('sum_q25', 'NA')}, Q75={profile.get('sum_q75', 'NA')}
- 과거 조합 합계 넓은 범위: Q10={profile.get('sum_q10', 'NA')}, Q90={profile.get('sum_q90', 'NA')}
- 과거 주요 패턴 빈도표:
{pattern_top_text}

[추천 조합 원본표]
{reco_text}

[조합별 수치 요약표]
{combo_detail_text}

[추천 조합에 포함된 개별 번호 수치표]
{number_detail_text}

[번호별 통계점수 상위 20개]
{top_text}

설명 형식:
1. 먼저 '분석 요약'을 2~3문장으로 작성.
2. 그 다음 추천 1~3순위 조합별로 아래 항목을 반드시 수치와 함께 설명.
   - 조합통계점수
   - 번호평균통계점수
   - 장기빈도점수평균
   - 최근흐름점수평균
   - 미출현간격점수평균
   - 전체당첨횟수합
   - 최근 {recent_window}회 출현합
   - 합계, 홀짝, 저고, 구간분포5, 소수개수, 3배수개수, 5배수개수, 연속번호쌍, 패턴종합점수
3. 마지막에는 '주의' 문단을 넣고 로또는 독립 무작위 추첨이라 이 분석이 당첨을 보장하지 않는다고 명확히 말해.
4. 숫자는 표에 있는 값을 그대로 사용하고, 없는 값은 추측하지 마.
"""
    try:
        response = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[
                {"role": "system", "content": "너는 복권 통계를 신중하고 수치적으로 설명하는 한국어 통계 분석 보조 AI다. 당첨 보장 표현은 금지한다."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.25,
            max_tokens=1800,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"NVIDIA API 호출 중 오류가 발생했습니다.\n\n오류 내용: {e}"


def parse_ai_combo_response(text: str) -> Tuple[List[Dict], Optional[str]]:
    """AI가 반환한 JSON에서 조합 추출"""
    if not text:
        return [], "응답이 비어 있습니다."
    candidates = []
    raw = text.strip()
    blocks = re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    parse_targets = blocks + [raw]
    for target in parse_targets:
        target = target.strip()
        try:
            obj = json.loads(target)
        except Exception:
            # 텍스트 중 배열 또는 객체 부분만 추출 시도
            m = re.search(r"(\{.*\}|\[.*\])", target, flags=re.DOTALL)
            if not m:
                continue
            try:
                obj = json.loads(m.group(1))
            except Exception:
                continue
        if isinstance(obj, dict):
            if isinstance(obj.get("recommendations"), list):
                candidates = obj["recommendations"]
            elif isinstance(obj.get("combinations"), list):
                candidates = obj["combinations"]
            elif isinstance(obj.get("조합"), list):
                candidates = obj["조합"]
            else:
                candidates = [obj]
        elif isinstance(obj, list):
            candidates = obj
        if candidates:
            break
    if not candidates:
        return [], "AI 응답에서 JSON 조합을 찾지 못했습니다."
    return candidates, None


def normalize_ai_combo_items(items: List[Dict]) -> List[Dict]:
    """AI 조합 응답을 검증 가능한 형태로 정리"""
    cleaned = []
    seen = set()
    for item in items:
        if isinstance(item, dict):
            nums = item.get("numbers") or item.get("번호") or item.get("combination") or item.get("조합")
            strategy = item.get("strategy") or item.get("전략") or "AI 통계 조합"
            reason = item.get("reason") or item.get("이유") or item.get("설명") or "AI가 통계표를 참고해 구성한 조합"
        elif isinstance(item, list):
            nums = item
            strategy = "AI 통계 조합"
            reason = "AI가 통계표를 참고해 구성한 조합"
        else:
            continue
        try:
            nums = sorted([int(x) for x in nums])
        except Exception:
            continue
        if len(nums) != 6 or len(set(nums)) != 6:
            continue
        if any(n < 1 or n > 45 for n in nums):
            continue
        t = tuple(nums)
        if t in seen:
            continue
        seen.add(t)
        cleaned.append({"numbers": nums, "strategy": str(strategy), "reason": str(reason)})
    return cleaned


def ai_items_to_reco_df(items: List[Dict], stats_df: pd.DataFrame, profile: Dict, recent_window: int) -> pd.DataFrame:
    score_map = stats_df.set_index("번호")["통계점수"].to_dict()
    rows = []
    for item in items:
        nums = sorted(item["numbers"])
        score, details = evaluate_statistical_combo(nums, score_map, profile)
        numeric_reason = build_combo_reason(nums, stats_df, recent_window)
        rows.append({
            "추천순위": 0,
            "번호1": nums[0],
            "번호2": nums[1],
            "번호3": nums[2],
            "번호4": nums[3],
            "번호5": nums[4],
            "번호6": nums[5],
            "조합통계점수": score,
            "합계": details["합계"],
            "홀짝": f"{details['홀수개수']}:{details['짝수개수']}",
            "저고": f"{details['저번호개수']}:{details['고번호개수']}",
            "구간분포5": details["5구간분포"],
            "소수개수": details["소수개수"],
            "3배수개수": details["3배수개수"],
            "5배수개수": details["5배수개수"],
            "연속번호쌍": details["연속번호쌍"],
            "끝자리최대중복": details["같은끝자리최대"],
            "패턴종합점수": details["패턴종합점수"],
            "패턴요약": details["패턴요약"],
            "AI전략": item.get("strategy", "AI 통계 조합"),
            "AI추천이유": item.get("reason", ""),
            "수치추천이유": numeric_reason,
        })
    df = pd.DataFrame(rows).sort_values("조합통계점수", ascending=False).reset_index(drop=True)
    if not df.empty:
        df["추천순위"] = range(1, len(df) + 1)
    return df


def ask_nvidia_make_recommendations(
    api_key: str,
    stats_df: pd.DataFrame,
    history_df: pd.DataFrame,
    profile: Dict,
    latest_draw: int,
    verification_status: str,
    recent_window: int,
    recommend_count: int,
    candidate_pool_size: int,
) -> Tuple[pd.DataFrame, str]:
    """AI가 통계표를 보고 직접 조합을 만들게 한 뒤, 앱에서 다시 검증/점수화"""
    if not api_key or api_key == "YOUR_API_KEY":
        return pd.DataFrame(), "NVIDIA API 키가 설정되지 않았습니다."

    top_df = stats_df.head(int(candidate_pool_size)).copy()
    top_cols = [
        "번호", "통계점수", "전체당첨횟수", "전체출현률", f"최근{recent_window}회출현횟수",
        f"최근{recent_window}회기대대비", "마지막출현후경과회차", "평균출현간격", "미출현간격비율",
        "장기빈도점수", "최근흐름점수", "미출현간격점수", "보너스점수",
    ]
    top_cols = [c for c in top_cols if c in top_df.columns]
    top_text = top_df[top_cols].to_string(index=False)
    pattern_top_text = profile.get("pattern_top_text", "패턴 빈도표 없음")

    recent_rows = []
    for _, row in history_df.sort_values("회차", ascending=False).head(15).iterrows():
        recent_rows.append({
            "회차": int(row["회차"]),
            "번호": extract_main_numbers(row.to_dict()),
            "보너스": int(row["보너스"]),
        })

    prompt = f"""
너는 로또 통계표를 읽고 참고용 조합을 구성하는 AI야.
당첨을 예측하거나 보장하면 안 된다. 통계적으로 균형 있는 참고 조합만 만들어라.

[조건]
- 만들어야 할 조합 수: {int(recommend_count)}개
- 각 조합은 1~45 사이 정수 6개
- 한 조합 안에서 중복 금지
- 가능하면 아래 통계점수 상위 {int(candidate_pool_size)}개 번호를 중심으로 구성
- 모든 조합이 너무 비슷하지 않게 분산
- 합계는 과거 IQR(Q25~Q75) 근처를 우선 고려하되, 무리하게 맞추지 않음
- 패턴은 홀짝/저고 2가지만 보지 말고, 아래 패턴 빈도표 전체를 참고
- 홀수만 있거나 저번호만 있는 극단 패턴도 실제 등장 빈도에 따라 판단
- 5구간 분포, 3구간 분포, 연속번호, 끝자리, 소수, 3의 배수, 5의 배수, 간격 패턴까지 함께 고려
- 단, 가장 흔한 패턴만 복사하지 말고 조합끼리는 서로 다르게 분산

[기본 정보]
최신 반영 회차: {latest_draw}회
데이터 검증 상태: {verification_status}
최근 흐름 분석 범위: 최근 {recent_window}회
과거 조합 합계 IQR: Q25={profile.get('sum_q25', 'NA')}, Q75={profile.get('sum_q75', 'NA')}
과거 주요 패턴 빈도표:
{pattern_top_text}

[최근 15회 결과]
{json.dumps(recent_rows, ensure_ascii=False)}

[번호별 통계점수 상위표]
{top_text}

반드시 아래 JSON 형식만 출력해라. JSON 밖에 설명 문장을 쓰지 마라.
{{
  "recommendations": [
    {{"numbers": [1, 2, 3, 4, 5, 6], "strategy": "전략명", "reason": "수치 기준을 반영한 짧은 이유"}}
  ]
}}
"""
    try:
        client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)
        response = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[
                {"role": "system", "content": "너는 JSON만 출력하는 통계 조합 생성기다. 당첨 보장 표현은 금지한다."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.55,
            max_tokens=1600,
        )
        raw_text = response.choices[0].message.content or ""
        items, err = parse_ai_combo_response(raw_text)
        if err:
            return pd.DataFrame(), err + "\n\n원본 응답:\n" + raw_text
        cleaned = normalize_ai_combo_items(items)
        if not cleaned:
            return pd.DataFrame(), "AI가 만든 조합이 검증 규칙을 통과하지 못했습니다. 다시 시도해보세요.\n\n원본 응답:\n" + raw_text
        df = ai_items_to_reco_df(cleaned, stats_df, profile, recent_window)
        return df, raw_text
    except Exception as e:
        return pd.DataFrame(), f"NVIDIA API 호출 중 오류가 발생했습니다.\n\n오류 내용: {e}"


def ask_nvidia_ai(api_key, generated_numbers, pool_df, latest_draw, verification_status):
    if not api_key or api_key == "YOUR_API_KEY":
        return "NVIDIA API 키가 설정되지 않았습니다. Streamlit Secrets에 NVIDIA_API_KEY를 넣거나 왼쪽 입력창에 키를 입력해주세요."

    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)
    top_numbers_text = pool_df.sort_values("당첨횟수", ascending=False).head(15).to_string(index=False)

    prompt = f"""
아래 로또 번호 데이터와 생성 번호를 한국어로 짧게 설명해줘.

최신 반영 회차: {latest_draw}회
데이터 검증 상태: {verification_status}
생성된 번호 조합: {generated_numbers}
생성 범위 내 당첨횟수 상위 일부:
{top_numbers_text}

필수 조건:
1. 당첨을 보장하는 표현 금지.
2. 로또는 매회 독립적인 무작위 추첨이라고 안내.
3. 데이터 검증 상태가 '공식 검증 통과'가 아니면, 데이터는 공식값과 최종 확인이 필요하다고 안내.
4. 5문장 이내.
"""

    try:
        response = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[
                {"role": "system", "content": "너는 복권 데이터를 신중하게 설명하는 한국어 AI 도우미다."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=600,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"NVIDIA API 호출 중 오류가 발생했습니다.\n\n오류 내용: {e}"


# =========================================================
# 화면
# =========================================================
st.title("🎲 통계 패턴 분석형 AI 로또 번호 분석/생성기 v8")
st.write("mirror 데이터를 빠르게 불러오고 내부 무결성 검사를 기본 적용합니다. 공식 API 검증은 동행복권 사이트가 클라우드 접속을 차단할 수 있어 선택 기능으로 제공합니다.")

st.warning(
    "로또는 무작위 추첨입니다. 과거 당첨횟수는 미래 당첨을 보장하지 않습니다. "
    "이 앱은 데이터 확인과 재미용 번호 생성 도구입니다."
)

with st.sidebar:
    st.header("⚙️ 설정")

    data_mode = st.radio(
        "데이터 불러오기 방식",
        [
            "GitHub mirror + 내부 무결성 검사 추천",
            "GitHub mirror + 공식 API 검증 시도 느림",
            "동행복권 공식 API만 사용 매우 느림",
        ],
        index=0,
        help="Streamlit Cloud에서는 동행복권 공식 API가 HTML 대기/차단 페이지를 반환할 수 있어, 기본은 빠른 mirror + 내부 검사로 설정했습니다.",
    )

    verify_count = st.slider(
        "공식 API와 비교할 최신 회차 수",
        min_value=1,
        max_value=10,
        value=3,
        help="공식 검증 시도 모드에서만 사용됩니다. 숫자가 클수록 느려질 수 있고, 공식 API가 막힌 환경에서는 검증 불가로 표시됩니다.",
    )

    st.divider()
    input_api_key = st.text_input(
        "NVIDIA API 키 선택 입력",
        value=NVIDIA_API_KEY if NVIDIA_API_KEY != "YOUR_API_KEY" else "",
        type="password",
        help="Streamlit Secrets에 넣었다면 비워도 됩니다.",
    )
    final_api_key = input_api_key or NVIDIA_API_KEY

    if st.button("NVIDIA API 키 상태 확인"):
        ok, status_msg = check_nvidia_api_status(final_api_key)
        if ok:
            st.success(status_msg)
        else:
            st.error(status_msg)

    with st.expander("Secrets 입력 형식 보기"):
        st.code('NVIDIA_API_KEY = "nvapi-여기에_새_API키"', language="toml")
        st.write("실행 중인 앱은 Streamlit Cloud Secrets를 자동으로 수정할 수 없습니다. 새 키를 발급받은 뒤 Streamlit 앱 Settings → Secrets에 직접 저장하고 Reboot하세요.")

    if st.button("앱 캐시 새로고침"):
        st.cache_data.clear()
        st.rerun()

    st.caption("GitHub에 실제 API 키를 올리지 말고 Streamlit Secrets를 사용하세요.")

load_error = None
history_df = pd.DataFrame()
source_note = ""

with st.spinner("로또 데이터를 불러오는 중입니다..."):
    if data_mode in ["GitHub mirror + 내부 무결성 검사 추천", "GitHub mirror + 공식 API 검증 시도 느림"]:
        history_df, load_error = fetch_mirror_all()
        if history_df.empty:
            latest_row, latest_err = fetch_mirror_latest()
            if latest_row:
                history_df, load_error = fetch_mirror_by_draws(int(latest_row["회차"]))
            else:
                load_error = load_error or latest_err
        source_note = "GitHub mirror 기반"
    else:
        latest_row, latest_err = fetch_mirror_latest()
        if latest_row:
            history_df, load_error = fetch_official_history_slow(int(latest_row["회차"]))
        else:
            load_error = f"최신 회차 확인 실패: {latest_err}"
        source_note = "동행복권 공식 API 기반"

if history_df.empty:
    st.error("데이터를 불러오지 못했습니다.")
    if load_error:
        st.code(load_error)
    st.stop()

history_df = history_df.sort_values("회차", ascending=True).reset_index(drop=True)
latest_draw = int(history_df["회차"].max())

integrity_ok, integrity_df = validate_history_df(history_df)

verification_status = "검증 안 함"
comparison_df = pd.DataFrame()
verified_draws = []
verify_error = None

if data_mode == "GitHub mirror + 공식 API 검증 시도 느림":
    with st.spinner(f"최신 {verify_count}개 회차를 공식 API와 비교하는 중입니다..."):
        verification_status, comparison_df, verified_draws, verify_error = verify_with_official(history_df, verify_count)
elif data_mode == "동행복권 공식 API만 사용 매우 느림":
    verification_status = "공식 API 원본 사용"
else:
    verification_status = "공식 검증 생략"

count_df = make_count_table(history_df)
pattern_summary_df, pattern_raw_df = make_pattern_analysis(history_df)

# 상태 카드
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("최신 반영 회차", f"{latest_draw}회")
with c2:
    st.metric("총 데이터 회차", f"{len(history_df):,}개")
with c3:
    st.metric("데이터 출처", source_note)
with c4:
    st.metric("공식 검증 상태", verification_status)

if integrity_ok:
    st.success("기본 무결성 검사 통과: 번호 범위, 중복, 보너스 번호, 회차 연속성을 확인했습니다.")
else:
    st.error("기본 무결성 검사에서 문제가 발견되었습니다. 아래 '검증 상태' 탭을 확인하세요.")

if verification_status == "공식 검증 통과":
    st.success(f"최신 {verify_count}개 회차가 동행복권 공식 조회값과 일치했습니다.")
elif verification_status == "공식값과 불일치 발견":
    st.error("mirror 데이터와 동행복권 공식 조회값이 다른 회차가 있습니다. 번호 생성 전 검증 상태 탭을 확인하세요.")
elif verification_status in ["검증 불가", "일부만 검증됨"]:
    st.warning("동행복권 공식 API가 Streamlit Cloud 접속에 HTML 대기/차단 페이지를 반환해 공식 검증을 완료하지 못했습니다. mirror 데이터가 틀렸다는 뜻은 아니며, 기본 무결성 검사와 통계 기능은 계속 사용할 수 있습니다.")
    if verify_error:
        with st.expander("공식 검증 실패 사유 보기"):
            st.code(verify_error)


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "✅ 검증 상태",
    "📊 번호별 당첨횟수",
    "📅 회차별 당첨번호",
    "🎲 번호 생성기",
    "🧩 패턴 분석",
    "📈 통계 추천 조합",
    "📥 데이터 다운로드",
])

with tab1:
    st.header("✅ 데이터 검증 상태")

    st.subheader("1. 기본 무결성 검사")
    st.dataframe(integrity_df, use_container_width=True, hide_index=True)

    st.subheader("2. 공식 API 비교 검사")
    if data_mode == "GitHub mirror + 공식 API 검증 시도 느림":
        st.write(f"최신 {verify_count}개 회차를 동행복권 공식 조회값과 비교했습니다.")
        if not comparison_df.empty:
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        else:
            st.info("비교 결과가 없습니다.")
    elif data_mode == "GitHub mirror + 내부 무결성 검사 추천":
        st.info("현재는 빠른 mirror 모드라 공식 비교 검증을 생략했습니다. 왼쪽 설정에서 'GitHub mirror + 공식 API 검증 시도 느림'을 선택하면 비교합니다. 단, Streamlit Cloud에서는 공식 API가 막혀 검증 불가가 뜰 수 있습니다.")
    else:
        st.info("공식 API만 사용 중입니다. 다만 전체 회차를 공식 API로 불러오는 방식은 매우 느릴 수 있습니다.")

    st.subheader("3. 데이터 해석 기준")
    st.write(
        "'공식 검증 통과'는 최신 일부 회차가 공식 조회값과 일치했다는 의미입니다. "
        "전체 회차를 100% 보증한다는 뜻은 아니지만, mirror 데이터를 무작정 사용하는 것보다 신뢰도를 높입니다."
    )

with tab2:
    st.header("📊 1~45 번호별 당첨횟수")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("당첨횟수 표")
        st.dataframe(count_df, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("번호순 당첨횟수 그래프")
        chart_df = count_df.sort_values("번호").set_index("번호")[["당첨횟수"]]
        st.bar_chart(chart_df)
    st.subheader("🔥 당첨횟수 상위 10개 번호")
    st.dataframe(count_df.head(10), use_container_width=True, hide_index=True)

with tab3:
    st.header("📅 회차별 당첨번호 보기")
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_draw = st.number_input(
            "보고 싶은 회차",
            min_value=1,
            max_value=latest_draw,
            value=latest_draw,
            step=1,
        )
    selected_row = history_df[history_df["회차"].astype(int) == int(selected_draw)]
    with col2:
        if not selected_row.empty:
            row = selected_row.iloc[0]
            nums = [int(row[f"번호{i}"]) for i in range(1, 7)]
            st.subheader(f"{int(selected_draw)}회 당첨번호")
            st.write(f"추첨일: {row['추첨일']}")
            st.markdown("### " + "  ".join([f"`{n}`" for n in nums]) + f"  + 보너스 `{int(row['보너스'])}`")
            if safe_int(row.get("1등당첨금"), 0) > 0:
                st.write(f"1등 당첨금: {format_money(row.get('1등당첨금'))}")
            if safe_int(row.get("1등당첨자수"), 0) > 0:
                st.write(f"1등 당첨자 수: {int(row.get('1등당첨자수'))}명")
        else:
            st.warning("해당 회차 데이터가 없습니다.")

    st.divider()
    st.subheader("전체 회차 목록")
    st.dataframe(history_df.sort_values("회차", ascending=False), use_container_width=True, hide_index=True)

with tab4:
    st.header("🎲 조건별 로또 번호 생성기")

    if verification_status in ["공식값과 불일치 발견", "검증 불가"]:
        st.warning("현재 데이터 검증 상태가 완전하지 않습니다. 번호 생성은 가능하지만, 데이터는 공식 사이트에서 최종 확인하세요.")

    mode = st.radio(
        "번호 생성 방식",
        [
            "전체 1~45에서 생성",
            "당첨횟수 N회 이상 번호에서만 생성",
            "당첨횟수 상위 N개 번호에서만 생성",
        ],
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        default_min = int(count_df["당첨횟수"].quantile(0.75))
        min_count = st.number_input(
            "N회 이상 기준",
            min_value=0,
            max_value=int(count_df["당첨횟수"].max()),
            value=default_min,
            step=1,
        )
    with col2:
        top_n = st.number_input("상위 N개 기준", min_value=6, max_value=45, value=20, step=1)
    with col3:
        set_count = st.number_input("생성할 조합 개수", min_value=1, max_value=20, value=5, step=1)

    use_weight = st.checkbox("당첨횟수가 높은 번호가 더 잘 뽑히도록 가중치 적용", value=False)

    if st.button("🎲 번호 생성하기", type="primary"):
        generated_numbers, pool_df = generate_lotto_numbers(
            count_df=count_df,
            mode=mode,
            min_count=int(min_count),
            top_n=int(top_n),
            use_weight=use_weight,
            set_count=int(set_count),
        )
        if len(generated_numbers) == 0:
            st.error("선택 가능한 번호가 6개보다 적습니다. 기준을 낮춰주세요.")
            st.dataframe(pool_df, use_container_width=True, hide_index=True)
        else:
            st.session_state["generated_numbers"] = generated_numbers
            st.session_state["pool_df"] = pool_df
            st.session_state["mode"] = mode

    if "generated_numbers" in st.session_state:
        st.subheader("생성된 번호")
        result_rows = []
        for i, nums in enumerate(st.session_state["generated_numbers"], start=1):
            result_rows.append({"조합": i, "번호1": nums[0], "번호2": nums[1], "번호3": nums[2], "번호4": nums[3], "번호5": nums[4], "번호6": nums[5]})
        st.dataframe(pd.DataFrame(result_rows), use_container_width=True, hide_index=True)

        st.subheader("현재 생성 범위에 포함된 번호")
        st.dataframe(st.session_state["pool_df"].sort_values("번호"), use_container_width=True, hide_index=True)

        if st.button("🤖 NVIDIA AI 코멘트 생성"):
            with st.spinner("NVIDIA AI가 설명을 생성하는 중입니다..."):
                ai_comment = ask_nvidia_ai(
                    api_key=final_api_key,
                    generated_numbers=st.session_state["generated_numbers"],
                    pool_df=st.session_state["pool_df"],
                    latest_draw=latest_draw,
                    verification_status=verification_status,
                )
            st.subheader("🤖 AI 코멘트")
            st.write(ai_comment)

with tab5:
    st.header("🧩 과거 당첨 조합 패턴 분석")
    st.write(
        "홀짝/저고만 보는 것이 아니라, 구간분포, 합계구간, 연속번호, 끝자리, 소수, 배수, 간격, "
        "전회차 재등장 수까지 여러 패턴을 뽑아 실제 과거 데이터에서 자주 등장한 순서로 보여줍니다."
    )

    if pattern_summary_df.empty:
        st.info("패턴 분석 데이터가 없습니다.")
    else:
        p1, p2 = st.columns([2, 1])
        with p1:
            selected_pattern_type = st.selectbox(
                "보고 싶은 패턴 종류",
                sorted(pattern_summary_df["패턴종류"].unique().tolist()),
            )
        with p2:
            pattern_top_n = st.number_input("상위 몇 개까지 보기", min_value=3, max_value=100, value=20, step=1)

        selected_pattern_df = (
            pattern_summary_df[pattern_summary_df["패턴종류"] == selected_pattern_type]
            .sort_values(["패턴순위", "등장횟수"], ascending=[True, False])
            .head(int(pattern_top_n))
        )
        st.subheader(f"{selected_pattern_type} 빈도순")
        st.dataframe(selected_pattern_df, use_container_width=True, hide_index=True)

        st.subheader("패턴 종류별 1위 요약")
        top_each_type_df = (
            pattern_summary_df[pattern_summary_df["패턴순위"] == 1]
            .sort_values("패턴종류")
            .reset_index(drop=True)
        )
        st.dataframe(top_each_type_df, use_container_width=True, hide_index=True)

        st.subheader("전체 패턴값 등장횟수 순위")
        st.caption("서로 다른 패턴 종류는 전체 회차 수가 조금 다를 수 있으므로, 해석할 때는 같은 패턴 종류 안에서 비교하는 것이 가장 정확합니다.")
        overall_pattern_df = pattern_summary_df.sort_values(["등장횟수", "비율(%)"], ascending=False).head(100)
        st.dataframe(overall_pattern_df, use_container_width=True, hide_index=True)

        with st.expander("패턴 종류 설명"):
            st.write(
                "- 홀짝 개수: 당첨번호 6개 중 홀수/짝수가 각각 몇 개인지 봅니다. 예: 홀수 3개 / 짝수 3개\n\n"
                "- 저번호/고번호 개수: 1~22를 저번호, 23~45를 고번호로 나눠 개수를 봅니다.\n\n"
                "- 5구간 분포: 1~9 / 10~19 / 20~29 / 30~39 / 40~45에 각각 몇 개가 들어갔는지 봅니다. 예: 1-2-1-1-1\n\n"
                "- 3구간 분포: 1~15 / 16~30 / 31~45에 각각 몇 개가 들어갔는지 봅니다.\n\n"
                "- 합계 구간: 번호 6개의 합계를 10단위 또는 20단위 구간으로 묶어 봅니다.\n\n"
                "- 연속번호/끝자리/소수/배수/간격 패턴: 조합의 모양이 과거에 얼마나 자주 나왔는지 보기 위한 보조 기준입니다.\n\n"
                "- 전회차 번호 재등장 개수: 바로 전 회차 당첨번호 중 몇 개가 다음 회차에 다시 등장했는지 봅니다."
            )

        with st.expander("회차별 원본 패턴 데이터 보기"):
            st.dataframe(pattern_raw_df.sort_values(["회차", "패턴종류"], ascending=[False, True]), use_container_width=True, hide_index=True)

with tab6:
    st.header("📈 최근 결과 + 통계 기반 추천 조합")
    st.warning(
        "이 기능은 '가장 당첨될 것 같은 번호'를 보장하는 기능이 아닙니다. "
        "장기 빈도, 최근 흐름, 미출현 간격, 보너스 출현, 조합 균형을 점수화해 참고용 조합을 추천합니다. "
        "로또는 매회 독립적인 무작위 추첨입니다."
    )

    if verification_status in ["공식값과 불일치 발견", "검증 불가"]:
        st.warning("현재 데이터 검증 상태가 완전하지 않습니다. 추천 조합 생성 전 최신 회차는 동행복권 공식 사이트에서 최종 확인하세요.")

    st.subheader("분석 기준 설정")
    c1, c2, c3 = st.columns(3)
    with c1:
        recent_window = st.slider(
            "최근 흐름을 볼 회차 수",
            min_value=10,
            max_value=min(200, len(history_df)),
            value=min(50, len(history_df)),
            step=10,
            help="최근 N회 안에서 많이 나온 번호를 최근 흐름 점수에 반영합니다.",
        )
    with c2:
        top_pool_size = st.slider(
            "추천 후보 번호 범위",
            min_value=10,
            max_value=45,
            value=30,
            step=1,
            help="통계점수 상위 몇 개 번호 안에서 조합 후보를 만들지 정합니다.",
        )
    with c3:
        recommend_count = st.number_input("추천 조합 개수", min_value=1, max_value=20, value=5, step=1)

    st.subheader("통계 가중치")
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        weight_total = st.slider("장기 빈도", 0, 100, 35)
    with w2:
        weight_recent = st.slider("최근 흐름", 0, 100, 30)
    with w3:
        weight_gap = st.slider("미출현 간격", 0, 100, 25)
    with w4:
        weight_bonus = st.slider("보너스 출현", 0, 100, 10)

    c4, c5 = st.columns(2)
    with c4:
        candidate_count = st.slider(
            "검토할 후보 조합 수",
            min_value=1000,
            max_value=20000,
            value=5000,
            step=1000,
            help="숫자가 클수록 더 많은 조합을 비교하지만 실행 시간이 조금 늘어납니다.",
        )
    with c5:
        include_top_six = st.checkbox("통계점수 상위 6개 조합도 후보에 포함", value=True)

    with st.expander("이 기능이 점수에 반영하는 정보"):
        st.write(
            "1. 장기 빈도: 전체 회차에서 많이 나온 번호인지 봅니다.\n\n"
            "2. 최근 흐름: 최근 N회 안에서 상대적으로 자주 나온 번호인지 봅니다.\n\n"
            "3. 미출현 간격: 평균 출현 간격 대비 최근에 얼마나 오래 안 나왔는지 봅니다.\n\n"
            "4. 보너스 출현: 보너스 번호로 자주 나온 정도를 약하게 반영합니다.\n\n"
            "5. 조합 패턴: 홀짝, 저고, 5구간/3구간 분포, 합계구간, 연속번호, 끝자리, 소수, 3의 배수, 5의 배수, 번호 간격 패턴을 함께 봅니다."
        )

    st.subheader("AI 직접 조합 생성 옵션")
    ai_c1, ai_c2 = st.columns(2)
    with ai_c1:
        ai_candidate_pool = st.slider(
            "AI가 참고할 통계점수 상위 번호 개수",
            min_value=12,
            max_value=45,
            value=28,
            step=1,
            help="AI에게 번호별 통계표를 보여줄 범위입니다. 너무 좁으면 조합이 비슷해질 수 있습니다.",
        )
    with ai_c2:
        ai_recommend_count = st.number_input("AI가 직접 만들 추천 조합 개수", min_value=1, max_value=10, value=5, step=1)

    if st.button("📈 통계 추천 조합 만들기", type="primary"):
        with st.spinner("여러 후보 조합을 만들고 통계 점수로 정렬하는 중입니다..."):
            reco_df, stat_score_df, combo_profile = generate_statistical_recommendations(
                history_df=history_df,
                count_df=count_df,
                recent_window=int(recent_window),
                weight_total=int(weight_total),
                weight_recent=int(weight_recent),
                weight_gap=int(weight_gap),
                weight_bonus=int(weight_bonus),
                candidate_count=int(candidate_count),
                recommend_count=int(recommend_count),
                top_pool_size=int(top_pool_size),
                include_top_six=bool(include_top_six),
            )
            st.session_state["reco_df"] = reco_df
            st.session_state["stat_score_df"] = stat_score_df
            st.session_state["stat_recent_window"] = int(recent_window)
            st.session_state["combo_profile"] = combo_profile

    if st.button("🤖 AI가 직접 추천 조합 만들기"):
        with st.spinner("NVIDIA AI가 통계표를 읽고 직접 조합을 만드는 중입니다..."):
            stat_score_df = make_statistical_score_table(
                history_df=history_df,
                count_df=count_df,
                recent_window=int(recent_window),
                weight_total=int(weight_total),
                weight_recent=int(weight_recent),
                weight_gap=int(weight_gap),
                weight_bonus=int(weight_bonus),
            )
            combo_profile = make_historical_combo_profile(history_df)
            ai_reco_df, ai_raw = ask_nvidia_make_recommendations(
                api_key=final_api_key,
                stats_df=stat_score_df,
                history_df=history_df,
                profile=combo_profile,
                latest_draw=latest_draw,
                verification_status=verification_status,
                recent_window=int(recent_window),
                recommend_count=int(ai_recommend_count),
                candidate_pool_size=int(ai_candidate_pool),
            )
            if ai_reco_df.empty:
                st.error(ai_raw)
            else:
                st.session_state["ai_reco_df"] = ai_reco_df
                st.session_state["ai_raw_response"] = ai_raw
                st.session_state["reco_df"] = ai_reco_df
                st.session_state["stat_score_df"] = stat_score_df
                st.session_state["stat_recent_window"] = int(recent_window)
                st.session_state["combo_profile"] = combo_profile
                st.success("AI가 만든 조합을 앱에서 다시 검증하고 통계점수로 정렬했습니다.")

    if "reco_df" in st.session_state:
        st.subheader("추천 조합")
        st.dataframe(st.session_state["reco_df"], use_container_width=True, hide_index=True)

        if "AI전략" in st.session_state["reco_df"].columns:
            with st.expander("AI 원본 응답 보기"):
                st.code(st.session_state.get("ai_raw_response", ""))

        st.subheader("조합별 수치 요약")
        numeric_combo_df = make_combo_numeric_detail_df(
            st.session_state["reco_df"],
            st.session_state["stat_score_df"],
            st.session_state["stat_recent_window"],
        )
        st.dataframe(numeric_combo_df, use_container_width=True, hide_index=True)

        with st.expander("추천 조합 개별 번호 수치 근거 보기"):
            numeric_number_df = make_number_numeric_detail_df(
                st.session_state["reco_df"],
                st.session_state["stat_score_df"],
                st.session_state["stat_recent_window"],
            )
            st.dataframe(numeric_number_df, use_container_width=True, hide_index=True)

        st.subheader("번호별 통계점수 상위 15개")
        st.dataframe(st.session_state["stat_score_df"].head(15), use_container_width=True, hide_index=True)

        with st.expander("추천 조합 기준값 보기"):
            profile = st.session_state.get("combo_profile", {})
            st.write(
                f"과거 조합 합계의 주요 범위: 약 {profile.get('sum_q25', 0):.0f} ~ {profile.get('sum_q75', 0):.0f}점, "
                f"넓은 범위: 약 {profile.get('sum_q10', 0):.0f} ~ {profile.get('sum_q90', 0):.0f}점"
            )
            st.write("과거 주요 패턴 빈도표 상위값:")
            st.text(profile.get("pattern_top_text", "패턴 빈도표 없음"))

        if st.button("🤖 통계 추천 AI 설명 생성"):
            with st.spinner("NVIDIA AI가 통계 추천 결과를 설명하는 중입니다..."):
                ai_comment = ask_nvidia_stat_ai(
                    api_key=final_api_key,
                    reco_df=st.session_state["reco_df"],
                    stats_df=st.session_state["stat_score_df"],
                    latest_draw=latest_draw,
                    verification_status=verification_status,
                    recent_window=st.session_state["stat_recent_window"],
                    combo_profile=st.session_state.get("combo_profile", {}),
                )
            st.subheader("🤖 수치형 통계 AI 설명")
            st.write(ai_comment)

with tab7:
    st.header("📥 데이터 다운로드")
    st.download_button(
        label="번호별 당첨횟수 CSV 다운로드",
        data=count_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="lotto_number_count_verified.csv",
        mime="text/csv",
    )
    st.download_button(
        label="회차별 당첨번호 CSV 다운로드",
        data=history_df.sort_values("회차", ascending=False).to_csv(index=False).encode("utf-8-sig"),
        file_name="lotto_history_verified.csv",
        mime="text/csv",
    )
    if not pattern_summary_df.empty:
        st.download_button(
            label="패턴 빈도표 CSV 다운로드",
            data=pattern_summary_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="lotto_pattern_summary.csv",
            mime="text/csv",
        )
    if not comparison_df.empty:
        st.download_button(
            label="공식 검증 비교결과 CSV 다운로드",
            data=comparison_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="lotto_official_verification.csv",
            mime="text/csv",
        )
