import json
import random
import socket
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI

# =========================================================
# NVIDIA API SETTINGS
# =========================================================
# Put your NVIDIA API key here.
# Example: NVIDIA_API_KEY = "nvapi-xxxxxxxxxxxxxxxx"
import os

try:
    NVIDIA_API_KEY = st.secrets.get("NVIDIA_API_KEY", "YOUR_API_KEY")
except Exception:
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "YOUR_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "minimaxai/minimax-m3"

# =========================================================
# LOTTO DATA SOURCES
# =========================================================
# Main source: GitHub Pages JSON mirror of Lotto 6/45 results.
# Fallback source: official DH Lottery per-draw API. The official site can return an HTML waiting/block page.
MIRROR_ALL_URL = "https://smok95.github.io/lotto/results/all.json"
MIRROR_LATEST_URL = "https://smok95.github.io/lotto/results/latest.json"
MIRROR_ONE_URL = "https://smok95.github.io/lotto/results/{draw_no}.json"
OFFICIAL_ONE_URL = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={draw_no}"
CACHE_FILE = Path("lotto_history_cache.csv")

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.dhlottery.co.kr/",
}

st.set_page_config(page_title="AI Lotto Number Generator", page_icon="🎲", layout="wide")


def get_local_ip():
    """Return local IP for phone access on the same Wi-Fi."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "IP check failed"


def safe_request_json(url: str, timeout: int = 15):
    """Request JSON. If HTML/waiting page is returned, raise a readable error."""
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()

    text = response.text.strip()
    if not text:
        raise ValueError(f"Empty response from {url}")

    first_char = text[:1]
    if first_char not in ["{", "["]:
        preview = text[:300].replace("\n", " ").replace("\r", " ")
        raise ValueError(
            "The server did not return JSON. It returned HTML/text instead. "
            f"Preview: {preview}"
        )

    try:
        return response.json()
    except json.JSONDecodeError:
        # Some servers prepend whitespace or odd characters. Try substring extraction.
        start_candidates = [i for i in [text.find("{"), text.find("[")] if i >= 0]
        if start_candidates:
            start = min(start_candidates)
            return json.loads(text[start:])
        raise


def normalize_one_record(item: dict):
    """Normalize mirror JSON or official JSON to one row."""
    if not isinstance(item, dict):
        return None

    # Mirror format
    if "draw_no" in item and "numbers" in item:
        numbers = item.get("numbers") or []
        if len(numbers) != 6:
            return None

        divisions = item.get("divisions") or []
        first_prize = None
        first_winners = None
        if divisions and isinstance(divisions, list) and len(divisions) > 0:
            first_prize = divisions[0].get("prize")
            first_winners = divisions[0].get("winners")

        raw_date = item.get("date", "")
        draw_date = raw_date[:10] if isinstance(raw_date, str) else ""

        return {
            "회차": int(item.get("draw_no")),
            "추첨일": draw_date,
            "번호1": int(numbers[0]),
            "번호2": int(numbers[1]),
            "번호3": int(numbers[2]),
            "번호4": int(numbers[3]),
            "번호5": int(numbers[4]),
            "번호6": int(numbers[5]),
            "보너스": int(item.get("bonus_no")),
            "1등당첨금": first_prize,
            "1등당첨자수": first_winners,
            "총판매금액": item.get("total_sales_amount"),
            "데이터출처": "GitHub mirror",
        }

    # Official DH Lottery format
    if item.get("returnValue") == "success" and "drwNo" in item:
        return {
            "회차": int(item.get("drwNo")),
            "추첨일": item.get("drwNoDate"),
            "번호1": int(item.get("drwtNo1")),
            "번호2": int(item.get("drwtNo2")),
            "번호3": int(item.get("drwtNo3")),
            "번호4": int(item.get("drwtNo4")),
            "번호5": int(item.get("drwtNo5")),
            "번호6": int(item.get("drwtNo6")),
            "보너스": int(item.get("bnusNo")),
            "1등당첨금": item.get("firstWinamnt"),
            "1등당첨자수": item.get("firstPrzwnerCo"),
            "총판매금액": item.get("totSellamnt"),
            "데이터출처": "DH Lottery official",
        }

    return None


def normalize_all_json(data):
    """Normalize all.json shape to a DataFrame."""
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            items = data["results"]
        elif "draws" in data and isinstance(data["draws"], list):
            items = data["draws"]
        elif "data" in data and isinstance(data["data"], list):
            items = data["data"]
        else:
            # Sometimes all.json can be a dict keyed by draw number.
            possible_values = list(data.values())
            if possible_values and all(isinstance(v, dict) for v in possible_values):
                items = possible_values
            else:
                raise ValueError("Unknown all.json format")
    else:
        raise ValueError("Unknown all.json format")

    rows = []
    for item in items:
        row = normalize_one_record(item)
        if row:
            rows.append(row)

    if not rows:
        raise ValueError("No valid lotto rows found in JSON")

    df = pd.DataFrame(rows).drop_duplicates(subset=["회차"]).sort_values("회차").reset_index(drop=True)
    return df


@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_from_mirror_all():
    data = safe_request_json(MIRROR_ALL_URL, timeout=20)
    return normalize_all_json(data)


@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_from_mirror_loop():
    latest = safe_request_json(MIRROR_LATEST_URL, timeout=20)
    latest_row = normalize_one_record(latest)
    if not latest_row:
        raise ValueError("Could not read latest draw number from mirror")

    latest_draw = int(latest_row["회차"])
    rows = []
    progress = st.progress(0, text="Downloading lotto history from mirror...")

    for draw_no in range(1, latest_draw + 1):
        try:
            item = safe_request_json(MIRROR_ONE_URL.format(draw_no=draw_no), timeout=10)
            row = normalize_one_record(item)
            if row:
                rows.append(row)
        except Exception:
            pass

        if draw_no % 10 == 0 or draw_no == latest_draw:
            progress.progress(draw_no / latest_draw, text=f"Downloading lotto history... {draw_no}/{latest_draw}")
        time.sleep(0.01)

    progress.empty()

    if not rows:
        raise ValueError("Mirror loop download failed")

    return pd.DataFrame(rows).drop_duplicates(subset=["회차"]).sort_values("회차").reset_index(drop=True)


@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_from_official_loop(max_draw: int = 1300):
    rows = []
    failed_streak = 0
    progress = st.progress(0, text="Trying official DH Lottery API...")

    for draw_no in range(1, max_draw + 1):
        try:
            item = safe_request_json(OFFICIAL_ONE_URL.format(draw_no=draw_no), timeout=10)
            row = normalize_one_record(item)
            if row:
                rows.append(row)
                failed_streak = 0
            else:
                failed_streak += 1
        except Exception:
            failed_streak += 1

        if draw_no % 10 == 0:
            progress.progress(min(draw_no / max_draw, 1.0), text=f"Trying official API... {draw_no}/{max_draw}")

        # After enough valid rows, 10 consecutive missing/failed draws means likely past latest draw.
        if len(rows) > 1000 and failed_streak >= 10:
            break

        time.sleep(0.03)

    progress.empty()

    if not rows:
        raise ValueError("Official DH Lottery API failed or returned non-JSON pages")

    return pd.DataFrame(rows).drop_duplicates(subset=["회차"]).sort_values("회차").reset_index(drop=True)


def save_cache(df: pd.DataFrame):
    df.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")


def load_cache():
    if not CACHE_FILE.exists():
        return None
    df = pd.read_csv(CACHE_FILE)
    required = {"회차", "번호1", "번호2", "번호3", "번호4", "번호5", "번호6", "보너스"}
    if not required.issubset(set(df.columns)):
        return None
    return df.sort_values("회차").reset_index(drop=True)


def load_lotto_history(force_refresh=False, source_mode="자동"):
    errors = []

    if not force_refresh:
        cached = load_cache()
        if cached is not None and not cached.empty:
            return cached, "캐시 파일", errors

    if source_mode in ["자동", "GitHub mirror"]:
        try:
            df = load_from_mirror_all()
            save_cache(df)
            return df, "GitHub mirror all.json", errors
        except Exception as e:
            errors.append(f"GitHub mirror all.json 실패: {e}")

        try:
            df = load_from_mirror_loop()
            save_cache(df)
            return df, "GitHub mirror per-draw", errors
        except Exception as e:
            errors.append(f"GitHub mirror per-draw 실패: {e}")

    if source_mode in ["자동", "동행복권 공식"]:
        try:
            df = load_from_official_loop()
            save_cache(df)
            return df, "동행복권 공식 API", errors
        except Exception as e:
            errors.append(f"동행복권 공식 API 실패: {e}")

    cached = load_cache()
    if cached is not None and not cached.empty:
        return cached, "캐시 파일", errors

    return pd.DataFrame(), "불러오기 실패", errors


def make_count_table(history_df: pd.DataFrame):
    main_counter = Counter()
    bonus_counter = Counter()

    for _, row in history_df.iterrows():
        for col in ["번호1", "번호2", "번호3", "번호4", "번호5", "번호6"]:
            if pd.notna(row[col]):
                main_counter[int(row[col])] += 1
        if pd.notna(row["보너스"]):
            bonus_counter[int(row["보너스"])] += 1

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
    weights = [max(float(w), 0.0) for w in weights]
    selected = []

    for _ in range(k):
        if not numbers:
            break
        total_weight = sum(weights)
        if total_weight <= 0:
            chosen = random.choice(numbers)
        else:
            r = random.uniform(0, total_weight)
            upto = 0
            chosen = numbers[-1]
            for number, weight in zip(numbers, weights):
                upto += weight
                if upto >= r:
                    chosen = number
                    break
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
            results.append(weighted_sample_without_replacement(pool_numbers, pool_df["당첨횟수"].tolist(), 6))
        else:
            results.append(sorted(random.sample(pool_numbers, 6)))
    return results, pool_df


def ask_nvidia_ai(api_key, generated_numbers, pool_df, latest_draw, source_name):
    if not api_key or api_key == "YOUR_API_KEY":
        return "NVIDIA API 키가 입력되지 않았습니다. 코드 상단의 NVIDIA_API_KEY 또는 왼쪽 사이드바 입력칸에 API 키를 넣어주세요."

    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)
    top_numbers_text = pool_df.sort_values("당첨횟수", ascending=False).head(15).to_string(index=False)

    prompt = f"""
너는 로또 번호 분석 보조 AI야.
아래 데이터는 로또 6/45 회차별 당첨번호를 바탕으로 계산한 것이다.

데이터 출처: {source_name}
최신 반영 회차: {latest_draw}회
생성된 번호 조합: {generated_numbers}
생성 범위의 당첨횟수 상위 번호:
{top_numbers_text}

요청:
1. 생성된 번호 조합을 짧게 설명해줘.
2. 많이 나온 번호 위주인지, 균형적인지 말해줘.
3. 과거 당첨횟수는 미래 당첨을 보장하지 않는다고 반드시 안내해줘.
4. 한국어로, 과몰입하지 않게 현실적으로 작성해줘.
"""

    try:
        response = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[
                {"role": "system", "content": "너는 복권 데이터를 설명하는 한국어 AI 도우미다. 절대 당첨을 보장하지 않는다."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=700,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"NVIDIA API 호출 중 오류가 발생했습니다.\n\n오류 내용: {e}"


# =========================================================
# UI
# =========================================================
st.title("🎲 AI 로또 번호 분석/생성기")
st.write("회차별 당첨번호 데이터를 불러와서 번호별 당첨횟수를 보고, 조건에 맞게 6개 번호를 생성합니다.")
st.warning("주의: 로또는 무작위 추첨입니다. 과거에 많이 나온 번호가 앞으로도 더 잘 나온다는 보장은 없습니다. 이 프로그램은 재미와 데이터 확인용으로만 사용하세요.")

with st.sidebar:
    st.header("⚙️ 설정")
    input_api_key = st.text_input("NVIDIA API 키", value=NVIDIA_API_KEY, type="password")

    st.divider()
    source_mode = st.radio("데이터 불러오기 방식", ["자동", "GitHub mirror", "동행복권 공식"], index=0)
    force_refresh = st.button("데이터 새로고침 / 캐시 다시 만들기")

    if st.button("캐시 파일 삭제"):
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
            st.success("캐시 파일을 삭제했습니다. 새로고침 버튼을 눌러 다시 불러오세요.")
        else:
            st.info("삭제할 캐시 파일이 없습니다.")

    st.divider()
    st.subheader("📱 스마트폰 접속")
    local_ip = get_local_ip()
    st.write("PC와 스마트폰이 같은 와이파이에 연결되어 있으면 아래 주소를 스마트폰 브라우저에 입력하세요.")
    st.code(f"http://{local_ip}:8501")
    st.caption("접속이 안 되면 run_phone.bat으로 실행하고, 윈도우 방화벽에서 허용을 누르세요.")

with st.spinner("로또 데이터를 불러오는 중입니다..."):
    history_df, source_name, load_errors = load_lotto_history(force_refresh=force_refresh, source_mode=source_mode)

if history_df.empty:
    st.error("로또 데이터를 불러오지 못했습니다.")
    if load_errors:
        st.write("오류 내용:")
        for err in load_errors:
            st.code(err)
    st.stop()

count_df = make_count_table(history_df)
latest_draw = int(history_df["회차"].max())
latest_date = history_df.loc[history_df["회차"].idxmax(), "추첨일"]

st.success(f"데이터 불러오기 완료: {source_name} / 최신 반영 {latest_draw}회 ({latest_date}) / 총 {len(history_df)}개 회차")
if load_errors:
    with st.expander("일부 데이터 소스 실패 기록 보기"):
        for err in load_errors:
            st.code(err)

tab1, tab2, tab3, tab4 = st.tabs(["📊 번호별 당첨횟수", "📅 회차별 당첨번호", "🎲 번호 생성기", "📥 다운로드"])

with tab1:
    st.header("📊 1~45 번호별 당첨횟수")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("당첨횟수 표")
        st.dataframe(count_df, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("당첨횟수 그래프")
        chart_df = count_df.sort_values("번호").set_index("번호")[["당첨횟수"]]
        st.bar_chart(chart_df)

    st.subheader("🔥 당첨횟수 상위 10개 번호")
    st.dataframe(count_df.head(10), use_container_width=True, hide_index=True)

with tab2:
    st.header("📅 회차별 당첨번호 보기")
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_draw = st.number_input("보고 싶은 회차", min_value=1, max_value=latest_draw, value=latest_draw, step=1)
    selected_row = history_df[history_df["회차"] == int(selected_draw)]
    if not selected_row.empty:
        row = selected_row.iloc[0]
        numbers = [int(row[f"번호{i}"]) for i in range(1, 7)]
        with col2:
            st.subheader(f"{int(selected_draw)}회 당첨번호")
            st.write(f"추첨일: {row['추첨일']}")
            st.markdown("### " + "  ".join([f"`{n}`" for n in numbers]) + f"  + 보너스 `{int(row['보너스'])}`")

    st.divider()
    st.subheader("전체 회차 목록")
    st.dataframe(history_df.sort_values("회차", ascending=False), use_container_width=True, hide_index=True)

with tab3:
    st.header("🎲 조건별 로또 번호 생성기")
    mode = st.radio("번호 생성 방식", ["전체 1~45에서 생성", "당첨횟수 N회 이상 번호에서만 생성", "당첨횟수 상위 N개 번호에서만 생성"])

    col1, col2, col3 = st.columns(3)
    with col1:
        default_min = int(count_df["당첨횟수"].quantile(0.75))
        min_count = st.number_input("N회 이상 기준", min_value=0, max_value=int(count_df["당첨횟수"].max()), value=default_min, step=1)
    with col2:
        top_n = st.number_input("상위 N개 기준", min_value=6, max_value=45, value=20, step=1)
    with col3:
        set_count = st.number_input("생성할 조합 개수", min_value=1, max_value=30, value=5, step=1)

    use_weight = st.checkbox("당첨횟수가 높은 번호가 더 잘 뽑히도록 가중치 적용", value=False)

    if st.button("🎲 번호 생성하기", type="primary"):
        generated_numbers, pool_df = generate_lotto_numbers(count_df, mode, int(min_count), int(top_n), use_weight, int(set_count))
        if not generated_numbers:
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
            with st.spinner("NVIDIA AI가 설명을 만드는 중입니다..."):
                ai_comment = ask_nvidia_ai(input_api_key, st.session_state["generated_numbers"], st.session_state["pool_df"], latest_draw, source_name)
            st.subheader("🤖 AI 코멘트")
            st.write(ai_comment)

with tab4:
    st.header("📥 데이터 다운로드")
    count_csv = count_df.to_csv(index=False).encode("utf-8-sig")
    history_csv = history_df.sort_values("회차", ascending=False).to_csv(index=False).encode("utf-8-sig")

    st.download_button("번호별 당첨횟수 CSV 다운로드", data=count_csv, file_name="lotto_number_count.csv", mime="text/csv")
    st.download_button("회차별 당첨번호 CSV 다운로드", data=history_csv, file_name="lotto_history.csv", mime="text/csv")
