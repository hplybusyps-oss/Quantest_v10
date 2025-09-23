import streamlit as st
import yfinance as yf
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mtick
from matplotlib.patches import Patch
from matplotlib import font_manager, rc
import numpy as np
import os
import pickle
from datetime import datetime, date



# --- session_state 초기화 ---
# 앱이 처음 실행되거나 새로고침될 때 'saved_results' 리스트가 없으면 만들어줍니다.
if 'saved_results' not in st.session_state:
    st.session_state.saved_results = []
    
# --- 웹/로컬 통합 한글 폰트 설정 ---

# 1. 폰트 파일의 경로를 설정합니다.
#    스크립트와 같은 폴더에 'malgun.ttf' 폰트 파일이 있어야 합니다.
font_name = 'malgun.ttf' 

# __file__은 현재 실행 중인 스크립트의 전체 경로를 의미합니다.
# 이를 통해 어떤 환경에서든 폰트 파일의 정확한 위치를 찾을 수 있습니다.
font_path = os.path.join(os.path.dirname(__file__), font_name)

# 2. 폰트 파일이 실제로 존재하는지 확인합니다.
if os.path.exists(font_path):
    # 3. Matplotlib의 폰트 목록에 해당 폰트를 추가합니다.
    fm.fontManager.addfont(font_path)
    
    # 4. 추가된 폰트를 Matplotlib의 기본 글꼴로 설정합니다.
    font_prop = fm.FontProperties(fname=font_path)
    plt.rc('font', family=font_prop.get_name())
else:
    # 폰트 파일이 없을 경우 경고 메시지를 출력하고, 시스템 기본 폰트를 시도합니다.
    print(f"경고: 폰트 파일 '{font_name}'을(를) 찾을 수 없습니다. 시스템 폰트를 사용합니다.")
    plt.rc('font', family='Malgun Gothic') # Windows 사용자를 위한 대비책

# 5. 마이너스 부호(-)가 네모로 깨지는 현상을 방지합니다.
plt.rc('axes', unicode_minus=False)     
# =============================================================================

# -----------------------------------------------------------------------------
# 1. GUI 화면 구성 (Streamlit)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="[Quantest] 퀀트 백테스트 프레임워크", page_icon="📈", layout="wide")

@st.cache_data
def load_Stock_list():
    try:
        # 프로그램(.exe 또는 .py)이 있는 폴더의 경로를 찾습니다.
        if getattr(sys, 'frozen', False):
            application_path = os.path.dirname(sys.executable)
        else:
            application_path = os.path.dirname(os.path.abspath(__file__))

        # 폴더 경로와 파일 이름을 합쳐 정확한 파일 경로를 만듭니다.
        csv_path = os.path.join(application_path, 'Stock_list.csv')

        # 완성된 경로를 이용해 CSV 파일을 읽습니다.
        df = pd.read_csv(csv_path, encoding='cp949')
        # --- 여기까지 ---
        df['display'] = df['Ticker'] + ' - ' + df['Name']
        return df
    except FileNotFoundError:
        st.error("'Stock_list.csv' 파일을 찾을 수 없습니다. 프로그램과 같은 폴더에 파일을 생성해주세요.")
        return None
    # UnicodeDecodeError에 대한 예외 처리 추가
    except UnicodeDecodeError:
        st.error("""
        'Stock_list.csv' 파일 인코딩 오류가 발생했습니다.
        파일을 열어 '다른 이름으로 저장' > 'CSV UTF-8' 형식으로 다시 저장해보세요.
        """)
        return None

etf_df = load_Stock_list()

st.sidebar.title("⚙️ 백테스트 설정")
st.sidebar.header("1. 기본 설정")

start_date = st.sidebar.date_input(
    "시작일",
    pd.to_datetime('2010-01-01').date() # 기본값을 date 객체로 명확히 변환
)
end_date = st.sidebar.date_input(
    "종료일",
    date.today() # 기본값으로 안정적인 date.today() 사용
)

# --- 통화 선택 UI를 제거하고, 나중에 티커 기반으로 자동 결정 ---

initial_capital = st.sidebar.number_input(
    "초기 투자금액",
    value=10000,
    min_value=0,
    step=1000, # 천 단위로 조절하기 쉽게 step 추가
    help="백테스트를 시작하는 초기 총 자산입니다. 통화는 선택된 자산군에 따라 자동 결정됩니다."
)
# 입력된 금액을 천 단위 쉼표로 포맷하여 바로 아래에 표시
st.sidebar.markdown(f"<p style='text-align: right; color: #555; margin-top: -10px; margin-bottom: 10px;'>{initial_capital:,.0f}</p>", unsafe_allow_html=True)


# 월별 추가 투자금액 입력
monthly_contribution = st.sidebar.number_input(
    "월별 추가 투자금액",
    value=1000, # 기본값을 1000으로 변경
    min_value=0,
    step=100, # 백 단위로 조절하기 쉽게 step 추가
    help="매월 리밸런싱 시점에 추가로 투자할 금액입니다."
)
# 입력된 금액을 천 단위 쉼표로 포맷하여 바로 아래에 표시
st.sidebar.markdown(f"<p style='text-align: right; color: #555; margin-top: -10px;'>{monthly_contribution:,.0f}</p>", unsafe_allow_html=True)


if etf_df is not None:
    # 드롭다운 목록을 생성합니다 ('티커 - 이름' 형식).
    benchmark_options = etf_df['display'].tolist()
    
    # 'SPY'에 해당하는 기본 선택값을 찾습니다.
    # 리스트에 'SPY'가 포함된 항목이 여러 개일 경우 첫 번째 항목을 사용합니다.
    default_benchmark = next((opt for opt in benchmark_options if 'SPY' in opt), benchmark_options[0])
    
    # st.selectbox를 사용하여 드롭다운 메뉴를 생성합니다.
    selected_benchmark_display = st.sidebar.selectbox(
        "벤치마크 선택", # 라벨을 "벤치마크 선택"으로 변경
        options=benchmark_options,
        index=benchmark_options.index(default_benchmark), # 'SPY'를 기본값으로 설정
        help="전략의 성과를 비교할 기준 지수(벤치마크)를 선택하세요."
    )
    # 선택된 값에서 실제 티커('SPY')만 추출합니다.
    benchmark_ticker = selected_benchmark_display.split(' - ')[0]
else:
    # Stock_list.csv 파일이 없는 경우, 기존의 텍스트 입력 방식을 유지합니다.
    benchmark_ticker = st.sidebar.text_input(
        "벤치마크 티커",
        value='SPY',
        help="전략의 성과를 비교하기 위한 기준 지수(벤치마크의 티커를 입력하세요."
)

st.sidebar.header("2. 실행 엔진 설정")
backtest_type = st.sidebar.radio(
    "백테스트 기준",
    ('일별', '월별'),
    index=0,
    help="""
    백테스트의 시간 단위를 결정합니다.
    - **일별**: 일별 데이터 사용
    - **월별**: 월별 데이터 사용
    """
)
rebalance_freq = st.sidebar.radio(
    "리밸런싱 주기",
    ('월별', '분기별'),
    index=0,
    help="포트폴리오의 자산 비중을 **재조정(리밸런싱)하는 주기**를 선택합니다."
)

# rebalance_day_help는 이미 가독성이 좋으므로 그대로 사용합니다.
rebalance_day_help = """
리밸런싱 시점을 결정합니다. 2월 포트폴리오를 결정하는 예시입니다.

**월말 기준:**
- **판단 시점:** 1월 31일 (1월 마지막 거래일)
- **사용 데이터:** 1월 31일까지의 모든 데이터
- **결과:** "1월의 성적표"를 보고 2월 계획을 짭니다. 가장 표준적인 방식입니다.

**월초 기준:**
- **판단 시점:** 2월 1일 (2월 첫 거래일)
- **사용 데이터:** 2월 1일까지의 모든 데이터
- **결과:** "2월 1일의 성적"까지 포함하여 2월 계획을 짭니다.
"""
rebalance_day = st.sidebar.radio("리밸런싱 기준일", ('월말', '월초'), index=0, help=rebalance_day_help)

transaction_cost = st.sidebar.slider(
    "거래 비용 (%)", 0.0, 1.0, 0.1, 0.01,
    help="매수 또는 매도 시 발생하는 **거래 비용(수수료, 슬리피지 등)을 시뮬레이션**합니다. 입력된 값은 편도(one-way) 기준입니다."
)
risk_free_rate = st.sidebar.slider(
    "무위험 수익률 (%)", 0.0, 5.0, 2.0, 0.1,
    help="**샤프 지수(Sharpe Ratio) 계산**에 사용되는 무위험 수익률입니다. 일반적으로 미국 단기 국채 금리를 사용하며, 연 수익률 기준으로 입력합니다."
)

# =============================================================================
#           [추가] 사이드바에 '티커 관리' 기능 추가
# =============================================================================
with st.sidebar.expander("티커 관리"):
    st.markdown("###### 현재 Stock_list.csv 내용")
    
    current_stocks_df = load_Stock_list()
    if current_stocks_df is not None and not current_stocks_df.empty:
        st.dataframe(current_stocks_df, height=100)

        # --- [추가] 티커 삭제 기능 ---
        st.markdown("---")
        st.markdown("###### 기존 티커 삭제")
        
        # 삭제할 티커를 선택하는 멀티셀렉트 박스
        tickers_to_delete = st.multiselect(
            "삭제할 티커를 선택하세요.",
            options=current_stocks_df['Ticker'].tolist()
        )
        
        if st.button("선택한 티커 삭제하기"):
            if tickers_to_delete:
                try:
                    # 1. 삭제할 티커를 제외한 나머지 데이터만 남깁니다.
                    updated_df = current_stocks_df[~current_stocks_df['Ticker'].isin(tickers_to_delete)]
                    
                    # 2. 파일 경로를 찾습니다.
                    if getattr(sys, 'frozen', False):
                        application_path = os.path.dirname(sys.executable)
                    else:
                        application_path = os.path.dirname(os.path.abspath(__file__))
                    csv_path = os.path.join(application_path, 'Stock_list.csv')

                    # 3. 수정된 데이터프레임을 CSV 파일에 덮어씁니다.
                    updated_df.to_csv(csv_path, index=False, encoding='cp949')
                    
                    st.success(f"{len(tickers_to_delete)}개의 티커를 삭제했습니다!")
                    
                    # 4. 변경사항을 즉시 반영합니다.
                    load_Stock_list.clear()
                    st.rerun()

                except Exception as e:
                    st.error(f"파일 수정 중 오류 발생: {e}")
            else:
                st.warning("삭제할 티커를 먼저 선택해주세요.")
        # --- 삭제 기능 끝 ---

    else:
        st.info("Stock_list.csv 파일이 비어있거나 찾을 수 없습니다.")

    st.markdown("---")
    st.markdown("###### 신규 티커 추가")

    with st.form(key='add_ticker_form', clear_on_submit=True):
        new_ticker = st.text_input("추가할 티커 (예: AAPL)").strip().upper()
        new_name = st.text_input("추가할 주식/ETF 이름 (예: Apple Inc)").strip()
        
        submitted = st.form_submit_button("티커 추가하기")
        if submitted:
            if new_ticker and new_name:
                # current_stocks_df가 None일 경우를 대비하여 빈 데이터프레임으로 초기화
                df_for_check = current_stocks_df if current_stocks_df is not None else pd.DataFrame(columns=['Ticker'])
                
                if new_ticker not in df_for_check['Ticker'].str.upper().values:
                    if getattr(sys, 'frozen', False):
                        application_path = os.path.dirname(sys.executable)
                    else:
                        application_path = os.path.dirname(os.path.abspath(__file__))
                    csv_path = os.path.join(application_path, 'Stock_list.csv')
                    
                    try:
                        import csv
                        # 파일이 없을 경우 헤더를 추가하기 위해 'a' 대신 'w' 모드와 os.path.exists를 확인
                        file_exists = os.path.exists(csv_path)
                        with open(csv_path, 'a', newline='', encoding='cp949') as f:
                            writer = csv.writer(f)
                            # 파일이 새로 생성되는 경우에만 헤더 작성
                            if not file_exists or os.path.getsize(csv_path) == 0:
                                writer.writerow(['Ticker', 'Name'])
                            writer.writerow([new_ticker, new_name])
                        
                        st.success(f"'{new_name}' ({new_ticker}) 추가 완료!")
                        load_Stock_list.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"파일 쓰기 중 오류 발생: {e}")
                else:
                    st.error(f"'{new_ticker}'는 이미 존재하는 티커입니다.")
            else:
                st.warning("티커와 이름을 모두 입력해주세요.")

st.sidebar.header("3. 자산군 설정")
if etf_df is not None:
    display_list = etf_df['display'].tolist()
    with st.sidebar.popover("카나리아 자산 선택하기", use_container_width=True):
        default_canary = [d for d in ['TIP - iShares TIPS Bond ETF'] if d in display_list]
        selected_canary_display = st.multiselect("카나리아 자산 검색", display_list, default=default_canary, label_visibility="collapsed")
    with st.sidebar.popover("공격 자산 선택하기", use_container_width=True):
        default_aggressive = [d for d in ['SPY - SPDR S&P 500 ETF Trust', 'IWM - iShares Russell 2000 ETF', 'VEA - Vanguard FTSE Developed Markets ETF', 'VWO - Vanguard FTSE Emerging Markets ETF', 'VNQ - Vanguard Real Estate ETF', 'DBC - Invesco DB Commodity Index Tracking Fund', 'IEF - iShares 7-10 Year Treasury Bond ETF', 'TLT - iShares 20+ Year Treasury Bond ETF'] if d in display_list]
        selected_aggressive_display = st.multiselect("공격 자산 검색", display_list, default=default_aggressive, label_visibility="collapsed")
    with st.sidebar.popover("방어 자산 선택하기", use_container_width=True):
        default_defensive = [d for d in ['BIL - SPDR Bloomberg 1-3 Month T-Bill ETF', 'IEF - iShares 7-10 Year Treasury Bond ETF'] if d in display_list]
        selected_defensive_display = st.multiselect("방어 자산 검색", display_list, default=default_defensive, label_visibility="collapsed")
    aggressive_tickers = [s.split(' - ')[0] for s in selected_aggressive_display]
    defensive_tickers = [s.split(' - ')[0] for s in selected_defensive_display]
    canary_tickers = [s.split(' - ')[0] for s in selected_canary_display]
    with st.sidebar.expander("✅ 선택된 자산 목록", expanded=True):
        st.markdown("**카나리아**"); st.info(f"{', '.join(canary_tickers) if canary_tickers else '없음'}")
        st.markdown("**공격**"); st.success(f"{', '.join(aggressive_tickers) if aggressive_tickers else '없음'}")
        st.markdown("**방어**"); st.warning(f"{', '.join(defensive_tickers) if defensive_tickers else '없음'}")
else:
    aggressive_tickers_str = st.sidebar.text_area("공격 자산군 (쉼표로 구분)", 'SPY,IWM,VEA,VWO,VNQ,DBC,IEF,TLT')
    defensive_tickers_str = st.sidebar.text_area("방어 자산군 (쉼표로 구분)", 'BIL,IEF')
    canary_tickers_str = st.sidebar.text_area("카나리아 자산 (쉼표로 구분)", 'TIP')
    aggressive_tickers = [t.strip().upper() for t in aggressive_tickers_str.split(',')]
    defensive_tickers = [t.strip().upper() for t in defensive_tickers_str.split(',')]
    canary_tickers = [t.strip().upper() for t in canary_tickers_str.split(',')]

st.sidebar.header("4. 시그널 설정")
momentum_type_help = """
- **13612U**: **1, 3, 6, 12개월** 수익률을 평균내어 안정적인 신호를 만듭니다. (HAA 전략 기본값)
- **평균 모멘텀**: 사용자가 **직접 입력한 기간들**의 수익률을 평균냅니다.
- **상대 모멘텀**: 여러 자산 중 특정 기간 동안 가장 많이 상승한 자산을 선택합니다. (상승장 추종에 유리)
"""
momentum_type = st.sidebar.selectbox("모멘텀 종류", ('13612U', '평균 모멘텀', '상대 모멘텀'), help=momentum_type_help)
momentum_periods_str = st.sidebar.text_input(
    "모멘텀 기간 (개월, 쉼표로 구분)", 
    value='1, 3, 6, 12', 
    help="""
    - **13612U**: 이 입력값은 **무시**됩니다.
    - **평균 모멘텀**: 사용할 기간을 쉼표로 구분하여 입력합니다. (예: 3, 6, 9)
    - **상대 모멘텀**: 입력된 숫자 중 **첫 번째 값**만 사용합니다. (예: '6' 입력 시 6개월 상대 모멘텀)
    """
)
st.sidebar.header("5. 포트폴리오 구성 전략")
use_canary = st.sidebar.toggle("카나리아 자산 사용 (Risk-On/Off)", value=True, help="체크 시, 카나리아 자산의 모멘텀이 양수일 때만 공격 자산에 투자합니다. 해제 시 항상 공격 자산군 내에서만 투자합니다.")
use_hybrid_protection = st.sidebar.toggle("하이브리드 보호 장치 사용", value=True, help="체크 시, 공격 자산으로 선택되었어도 개별 모멘텀이 음수이면 안전 자산으로 교체합니다.")
top_n_aggressive = st.sidebar.number_input("공격 자산 Top N", min_value=1, max_value=10, value=4, help="공격 자산군에서 모멘텀 순위가 높은 상위 N개의 자산을 선택합니다.")
top_n_defensive = st.sidebar.number_input("방어 자산 Top N", min_value=1, max_value=10, value=1, help="방어 자산군에서 모멘텀 순위가 높은 상위 N개의 자산을 선택합니다.")
weighting_scheme = st.sidebar.selectbox("자산 배분 방식", ('동일 비중 (Equal Weight)',), help="선택된 자산들에 어떤 비중으로 투자할지 결정합니다. (추후 확장 가능)")

# 모멘텀 기간 문자열을 숫자리스트로 변환하는 로직을 사이드바 영역으로 이동
try:
    momentum_periods = [int(p.strip()) for p in momentum_periods_str.split(',')]
except (ValueError, AttributeError):
    # 유효하지 않은 값이 입력된 경우, 에러 대신 기본값이나 빈 리스트로 처리
    momentum_periods = [1, 3, 6, 12] 

# 현재 사이드바 설정들을 딕셔너리로 모으는 함수
def gather_current_config():
    return {
        'start_date': start_date, 'end_date': end_date, 'initial_capital': initial_capital,
        'monthly_contribution': monthly_contribution, 'benchmark': benchmark_ticker,
        'backtest_type': backtest_type, 'rebalance_freq': rebalance_freq, 'rebalance_day': rebalance_day,
        'transaction_cost': transaction_cost / 100, 'risk_free_rate': risk_free_rate / 100,
        'tickers': {'AGGRESSIVE': aggressive_tickers, 'DEFENSIVE': defensive_tickers, 'CANARY': canary_tickers},
        'momentum_params': {'type': momentum_type, 'periods': momentum_periods},
        'portfolio_params': {'use_canary': use_canary, 'use_hybrid_protection': use_hybrid_protection,
                             'top_n_aggressive': top_n_aggressive, 'top_n_defensive': top_n_defensive,
                             'weighting': weighting_scheme}
    }

# 앱이 재실행될 때마다 현재 설정을 가져옴
current_config = gather_current_config()

# 마지막 실행 설정이 있고, 현재 설정과 다를 경우 '변경됨' 플래그를 True로 설정
if 'last_run_config' in st.session_state:
    settings_are_different = (st.session_state.last_run_config != current_config)
    st.session_state.settings_changed = settings_are_different

    # 설정이 변경되었고, 아직 토스트 알림을 보여주지 않았다면
    if settings_are_different and not st.session_state.get('toast_shown', False):
        st.toast("⚙️ 설정이 변경되었습니다!", icon="💡")
        st.session_state.toast_shown = True # 알림을 보여줬다고 기록
else:
    st.session_state.settings_changed = False

# --- 사이드바 설정 변경 감지 로직 끝 ---

# -----------------------------------------------------------------------------
# 2. 백엔드 로직 (데이터 처리 및 백테스트)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_price_data(tickers, start, end):
    try:
        raw_data = yf.download(tickers, start=start, end=end, progress=False)
        if raw_data.empty: st.error("데이터를 다운로드하지 못했습니다."); return None, None, None

        prices = raw_data['Close'].copy()
        prices.dropna(axis=0, how='all', inplace=True)
        
        successful_tickers = [t for t in tickers if t in prices.columns and not prices[t].isnull().all()]
        failed_tickers = [t for t in tickers if t not in successful_tickers]

        # --- [수정] 가장 늦게 시작하는 '핵심 원인' 티커 목록을 찾는 로직 ---
        if not successful_tickers:
            return pd.DataFrame(), failed_tickers, []

        start_dates = {ticker: prices[ticker].first_valid_index() for ticker in successful_tickers}
        
        valid_start_dates = [d for d in start_dates.values() if pd.notna(d)]
        if not valid_start_dates:
            return prices[successful_tickers].dropna(axis=0, how='any'), failed_tickers, []

        actual_latest_start = max(valid_start_dates)
        
        # 가장 늦은 날짜에 시작하는 모든 티커를 찾습니다.
        culprit_tickers = [ticker for ticker, date in start_dates.items() if date == actual_latest_start]
        
        # 사용자가 요청한 날짜보다 실제 시작일이 늦은 경우에만 "culprit"으로 간주합니다.
        # (주말이나 휴일 때문에 하루 이틀 늦게 시작하는 경우는 culprit으로 보지 않음)
        user_start_date_first_trading_day = prices.index[0]
        if actual_latest_start <= user_start_date_first_trading_day:
             culprit_tickers = []

        final_prices = prices[successful_tickers].dropna(axis=0, how='any')

        return final_prices, failed_tickers, culprit_tickers
    except Exception as e:
        st.error(f"데이터 다운로드 중 오류 발생: {e}"); return None, None, None

def calculate_cumulative_returns_with_dca(returns_series, initial_capital, monthly_contribution, contribution_dates):
    """적립식 투자를 반영하여 누적 자산 가치를 계산하는 함수"""
    portfolio_values = []
    current_capital = initial_capital

    # 기여금 날짜를 빠르게 조회하기 위해 set으로 변환
    contribution_dates_set = set(contribution_dates)

    for date, ret in returns_series.items():
        # 수익률에 따라 자산 가치 업데이트
        current_capital *= (1 + ret)
        
        # 추가 투자일인 경우, 해당 월의 추가 투자금 입금
        if date in contribution_dates_set and monthly_contribution > 0:
            current_capital += monthly_contribution
            
        portfolio_values.append(current_capital)
    
    return pd.Series(portfolio_values, index=returns_series.index)

# --- 👇 [신규 추가] 그래프용 전체 기간 모멘텀 계산 함수 ---
def calculate_full_momentum(prices, config):
    """그래프 표시를 위해 전체 기간에 대한 모멘텀 점수를 계산하는 함수"""
    mom_type = config['momentum_params']['type']
    
    if mom_type == '13612U':
        mom_periods = [1, 3, 6, 12]
    else:
        mom_periods = config['momentum_params'].get('periods', [1, 3, 6, 12])

    # 각 기간별 수익률을 계산 (근사치: 1개월 ≈ 21 거래일)
    returns_dfs = []
    for month in mom_periods:
        # shift를 사용하여 과거 가격 대비 수익률 계산
        returns_dfs.append(prices.pct_change(periods=month * 21).fillna(0))
        
    # 모든 기간의 수익률을 합산하여 평균
    if not returns_dfs:
        return pd.DataFrame(0, index=prices.index, columns=prices.columns)
        
    full_momentum_scores = sum(returns_dfs) / len(returns_dfs)
    return full_momentum_scores

def calculate_signals(prices, config):
    prices_copy = prices.copy()
    day_option = 'last' if config['rebalance_day'] == '월말' else 'first'
    if config['rebalance_freq'] == '분기별':
        prices_copy['year_quarter'] = prices_copy.index.to_period('Q').strftime('%Y-Q%q')
        rebal_dates = prices_copy.drop_duplicates('year_quarter', keep=day_option).index
    else: # 월별
        prices_copy['year_month'] = prices_copy.index.strftime('%Y-%m')
        rebal_dates = prices_copy.drop_duplicates('year_month', keep=day_option).index

    momentum_scores = pd.DataFrame(index=rebal_dates, columns=prices.columns)
    mom_type = config['momentum_params']['type']

    # --- CHANGED: '13612U' 선택 시 기간을 고정하도록 수정 ---
    if mom_type == '13612U':
        mom_periods = [1, 3, 6, 12]
    else:
        mom_periods = config['momentum_params']['periods']

    # --- CHANGED: '13612U'와 '평균 모멘텀' 로직 통합 및 '절대 모멘텀' 삭제 ---
    if mom_type in ['13612U', '평균 모멘텀']:
        for date in rebal_dates:
            returns = []
            for month in mom_periods:
                past_date = date - pd.DateOffset(months=month)
                if past_date < prices.index[0]:
                    returns.append(pd.Series(0.0, index=prices.columns))
                    continue
                past_price_idx = prices.index.get_indexer([past_date], method='nearest')[0]
                returns.append(prices.loc[date] / prices.iloc[past_price_idx] - 1)
            if returns:
                valid_returns = [r for r in returns if not r.empty]
                if valid_returns: momentum_scores.loc[date] = sum(valid_returns) / len(valid_returns)
                else: momentum_scores.loc[date] = 0.0
    
    elif mom_type == '상대 모멘텀':
        if not mom_periods: st.error("모멘텀 기간이 설정되지 않았습니다."); return pd.DataFrame()
        period_days = mom_periods[0] * 21 
        momentum_scores = prices.pct_change(periods=period_days)
        momentum_scores = momentum_scores.loc[rebal_dates].fillna(0)
            
    return momentum_scores.astype(float)

def construct_portfolio(momentum_scores, config, successful_tickers):
    canary_assets = [t for t in config['tickers']['CANARY'] if t in successful_tickers]
    aggressive_assets = [t for t in config['tickers']['AGGRESSIVE'] if t in successful_tickers]
    defensive_assets = [t for t in config['tickers']['DEFENSIVE'] if t in successful_tickers]
    params = config['portfolio_params']
    target_weights = pd.DataFrame(index=momentum_scores.index, columns=momentum_scores.columns).fillna(0.0)
    investment_mode = pd.Series(index=momentum_scores.index, dtype=str)

    for date in momentum_scores.index:
        best_defensive_assets = []
        if defensive_assets:
            best_defensive_scores = momentum_scores.loc[date, defensive_assets].dropna()
            if not best_defensive_scores.empty:
                best_defensive_assets = best_defensive_scores.nlargest(params['top_n_defensive']).index.tolist()
        
        is_risk_on = True
        if params['use_canary'] and canary_assets:
            canary_score = momentum_scores.loc[date, canary_assets].mean()
            if canary_score <= 0: is_risk_on = False

        if is_risk_on:
            investment_mode.loc[date] = 'Aggressive'
            top_aggressive_assets = momentum_scores.loc[date, aggressive_assets].dropna().nlargest(params['top_n_aggressive'])
            if not top_aggressive_assets.empty:
                weight_per_asset = 1.0 / len(top_aggressive_assets)
                for asset in top_aggressive_assets.index:
                    if params['use_hybrid_protection'] and momentum_scores.loc[date, asset] <= 0:
                        if best_defensive_assets:
                            for def_asset in best_defensive_assets:
                                target_weights.loc[date, def_asset] += weight_per_asset / len(best_defensive_assets)
                    else:
                        target_weights.loc[date, asset] = weight_per_asset
            else: 
                investment_mode.loc[date] = 'Defensive'
                if best_defensive_assets:
                    for def_asset in best_defensive_assets:
                        target_weights.loc[date, def_asset] = 1.0 / len(best_defensive_assets)
        else:
            investment_mode.loc[date] = 'Defensive'
            if best_defensive_assets:
                for def_asset in best_defensive_assets:
                    target_weights.loc[date, def_asset] = 1.0 / len(best_defensive_assets)
                
    return target_weights, investment_mode

def get_mdd_details(series):
    rolling_max = series.cummax()
    drawdown = (series - rolling_max) / rolling_max
    mdd_value = drawdown.min()
    mdd_end_date = drawdown.idxmin()
    pre_trough_series = series.loc[:mdd_end_date]
    mdd_start_date = pre_trough_series.idxmax()
    return mdd_value, mdd_start_date, mdd_end_date

def format_large_number(num, symbol='$'):
    """금액의 크기에 따라 K, M, B 단위를 붙여주는 함수"""
    if abs(num) >= 1_000_000_000:
        return f"{symbol}{num / 1_000_000_000:.1f}B"
    elif abs(num) >= 1_000_000:
        return f"{symbol}{num / 1_000_000:.1f}M"
    elif abs(num) >= 1_000:
        return f"{symbol}{num / 1_000:.1f}K"
    else:
        return f"{symbol}{num:,.0f}"
    

def get_saved_results(directory="backtest_results"):
    """저장된 결과 파일 목록과 표시용 이름을 반환하는 함수"""
    if not os.path.exists(directory) or not os.listdir(directory):
        return {} # 반환값을 딕셔너리로 통일
        
    file_list = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    
    results_map = {}
    for f in sorted(file_list, reverse=True):
        try:
            # --- 이 부분을 수정합니다 (날짜 포맷 일치) ---
            parts = f.replace('.pkl', '').split('_', 1)
            date_str = datetime.strptime(parts[0], '%Y%m%d%H%M%S').strftime('%Y-%m-%d')
            name = parts[1]
            display_name = f"{name} ({date_str})"
            results_map[f] = display_name
        except (IndexError, ValueError):
            continue
            
    return results_map

# -----------------------------------------------------------------------------
# 3. 메인 화면 구성 및 백테스트 실행
# -----------------------------------------------------------------------------
st.markdown("<a id='top'></a>", unsafe_allow_html=True)


st.title("📈 [Quantest] 퀀트 백테스트 프레임워크_v1.0")

# session_state에 표시할 토스트 메시지가 저장되어 있는지 확인합니다.
if 'toast_message' in st.session_state:
    # 메시지를 화면에 표시합니다.
    st.toast(st.session_state.toast_message, icon="💾")
    # 메시지를 표시한 후에는 다시 표시되지 않도록 session_state에서 삭제합니다.
    del st.session_state.toast_message

run_button_clicked = st.button("백테스트 실행", type="primary")
if st.session_state.get('settings_changed', False) and not run_button_clicked:
    st.warning("⚙️ 설정이 변경되었습니다. 백테스트를 다시 실행하여 최신 결과를 확인하세요!")

# '백테스트 실행' 버튼을 다시 생성하고, 모든 계산/실행 로직을 이 버튼 안으로 이동
if run_button_clicked:
    # 2. 상태 업데이트 로직을 블록의 맨 앞으로 이동
    #    이렇게 하면 이 블록이 실행되는 즉시 '변경됨' 상태가 해제됩니다.
    st.session_state.last_run_config = current_config # 사이드바에서 이미 만든 current_config 사용
    st.session_state.settings_changed = False
    st.session_state.toast_shown = False
    
    # --- 여기서부터는 기존의 백테스트 실행 코드와 거의 동일합니다 ---
    
    # config 변수를 current_config로 대체하거나 그대로 사용
    config = current_config 
    
    all_tickers = sorted(list(set(aggressive_tickers + defensive_tickers + canary_tickers + [benchmark_ticker])))
    
    if any(ticker.endswith('.KS') for ticker in all_tickers):
        currency_symbol = '₩'
    else:
        currency_symbol = '$'
    
    
    with st.spinner('데이터 로딩 및 백테스트 실행 중...'):
        prices, failed_tickers, culprit_tickers = get_price_data(all_tickers, config['start_date'], config['end_date'])
        
        if prices is None:
            st.error("데이터 로딩에 실패하여 백테스트를 중단합니다.")
            st.stop()

        momentum_scores = calculate_signals(prices, config)
        if momentum_scores.empty: st.error("모멘텀 시그널 계산에 실패했습니다."); st.stop()
        
        target_weights, investment_mode = construct_portfolio(momentum_scores, config, prices.columns.tolist())
        
        returns_freq = config['backtest_type'].split(' ')[0]
        if returns_freq == '월별':
            rebal_dates = momentum_scores.index
            prices_rebal = prices.loc[rebal_dates]
            returns_rebal = prices_rebal.pct_change()
            turnover = (target_weights.shift(1) - target_weights).abs().sum(axis=1) / 2
            costs = turnover * config['transaction_cost']
            portfolio_returns = (target_weights.shift(1) * returns_rebal).sum(axis=1) - costs
            portfolio_returns = portfolio_returns.fillna(0)
            benchmark_returns = returns_rebal[config['benchmark']].fillna(0)
        else: # 일별
            daily_weights = target_weights.reindex(prices.index, method='ffill').fillna(0)
            rebal_dates_series = pd.Series(index=prices.index, data=False)
            rebal_dates_series.loc[target_weights.index] = True
            turnover = (daily_weights.shift(1) - daily_weights).abs().sum(axis=1) / 2
            costs = turnover * config['transaction_cost']
            daily_returns = prices.pct_change().fillna(0)
            portfolio_returns = (daily_weights.shift(1) * daily_returns).sum(axis=1) - costs.where(rebal_dates_series, 0)
            benchmark_returns = daily_returns[config['benchmark']]

        contribution_dates = target_weights.index
        cumulative_returns = calculate_cumulative_returns_with_dca(portfolio_returns, config['initial_capital'], config['monthly_contribution'], contribution_dates)
        benchmark_cumulative = calculate_cumulative_returns_with_dca(benchmark_returns, config['initial_capital'], config['monthly_contribution'], contribution_dates)
        
        initial_cap = config['initial_capital']
        strategy_growth = (1 + portfolio_returns).cumprod() * initial_cap
        benchmark_growth = (1 + benchmark_returns).cumprod() * initial_cap

        strategy_dd = (strategy_growth / strategy_growth.cummax() - 1)
        benchmark_dd = (benchmark_growth / benchmark_growth.cummax() - 1)
                
        first_valid_date = cumulative_returns.first_valid_index()
        years = (cumulative_returns.index[-1] - first_valid_date).days / 365.25 if first_valid_date is not None else 0
        
        cagr, bm_cagr, mdd, bm_mdd, volatility, bm_volatility, sharpe_ratio, bm_sharpe_ratio, win_rate, bm_win_rate = (0,)*10
        if years > 0:
            cagr = (strategy_growth.iloc[-1]/initial_cap)**(1/years) - 1
            bm_cagr = (benchmark_growth.iloc[-1]/initial_cap)**(1/years) - 1
            mdd, mdd_start, mdd_end = get_mdd_details(strategy_growth)
            bm_mdd, bm_mdd_start, bm_mdd_end = get_mdd_details(benchmark_growth)
            trading_periods = 12 if returns_freq == '월별' else 252
            rf_rate = config['risk_free_rate']
            volatility = portfolio_returns.std() * np.sqrt(trading_periods)
            bm_volatility = benchmark_returns.std() * np.sqrt(trading_periods)
            sharpe_ratio = (cagr - rf_rate) / volatility if volatility != 0 else 0
            bm_sharpe_ratio = (bm_cagr - rf_rate) / bm_volatility if bm_volatility != 0 else 0
            win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns) if len(portfolio_returns) > 0 else 0
            bm_win_rate = (benchmark_returns > 0).sum() / len(benchmark_returns) if len(benchmark_returns) > 0 else 0

        total_months = len(target_weights.index)
        num_contributions = total_months - 1 if total_months > 0 else 0
        
        st.session_state['results'] = {
            'prices': prices, 'failed_tickers': failed_tickers, 'culprit_tickers': culprit_tickers,

            'config': config, 'currency_symbol': currency_symbol, 'etf_df': etf_df,
            'momentum_scores': momentum_scores,
            'timeseries': {
                'portfolio_value': cumulative_returns,
                'benchmark_value': benchmark_cumulative,
                'strategy_growth': strategy_growth,
                'benchmark_growth': benchmark_growth,
                'strategy_drawdown': strategy_dd,
                'benchmark_drawdown': benchmark_dd
            },
            'investment_mode': investment_mode, 'target_weights': target_weights, 'initial_cap': initial_cap,
            'metrics': {
                'final_assets': cumulative_returns.iloc[-1],
                'total_contribution': config['initial_capital'] + (config['monthly_contribution'] * num_contributions),
                'total_profit': cumulative_returns.iloc[-1] - (config['initial_capital'] + (config['monthly_contribution'] * num_contributions)),
                'cagr': cagr, 'mdd': mdd, 'mdd_start': mdd_start, 'mdd_end': mdd_end,
                'volatility': volatility, 'sharpe_ratio': sharpe_ratio, 'win_rate': win_rate,
                'bm_final_assets': benchmark_cumulative.iloc[-1],
                'bm_total_contribution': config['initial_capital'] + (config['monthly_contribution'] * num_contributions),
                'bm_total_profit': benchmark_cumulative.iloc[-1] - (config['initial_capital'] + (config['monthly_contribution'] * num_contributions)),
                'bm_cagr': bm_cagr, 'bm_mdd': bm_mdd, 'bm_mdd_start': bm_mdd_start, 'bm_mdd_end': bm_mdd_end,
                'bm_volatility': bm_volatility, 'bm_sharpe_ratio': bm_sharpe_ratio, 'bm_win_rate': bm_win_rate,
            },
            'portfolio_returns': portfolio_returns,
            'benchmark_returns': benchmark_returns
        }
        
        if 'backtest_save_name' in st.session_state:
            del st.session_state.backtest_save_name
        
        # 1. 현재 설정을 '마지막 실행 설정'으로 저장합니다.
        st.session_state.last_run_config = config
        # 2. '변경됨' 상태와 '토스트 표시' 상태를 모두 False로 초기화합니다.
        st.session_state.settings_changed = False
        st.session_state.toast_shown = False       
        st.session_state.result_selector = "--- 새로운 백테스트 실행 ---"

    # if 'last_uploaded_file_id' in st.session_state:
    #     del st.session_state['last_uploaded_file_id']

    st.rerun()        

# --- 탭과 결과 표시는 '백테스트 실행' 버튼 블록 바깥에 위치 ---
tab1, tab2 = st.tabs(["🚀 새로운 백테스트 결과", "📊 저장된 결과 비교"])

with tab1:
    st.header("🚀 백테스트 결과")
    st.divider()

    # --- .pkl 파일 업로드 기능 ---
    st.subheader("저장된 .pkl 파일 결과 보기")
    uploaded_file_tab1 = st.file_uploader(
        "상세 결과를 보고 싶은 .pkl 파일을 업로드하세요.",
        type=['pkl'],
        key="uploader_tab1"
    )

    if uploaded_file_tab1 is not None:
        # 파일 이름과 크기를 조합하여 현재 업로드된 파일을 식별하는 고유 ID를 생성합니다.
        current_file_id = f"{uploaded_file_tab1.name}-{uploaded_file_tab1.size}"
        
        # 이전에 기록된 파일 ID와 다를 경우, 즉 '새로운 파일'이 업로드된 경우에만 로직을 실행합니다.
        # 이렇게 하면 텍스트 입력 후 엔터를 치는 등의 단순 새로고침에서는 파일이 다시 로드되지 않습니다.
        if current_file_id != st.session_state.get('last_uploaded_file_id'):
            try:
                loaded_data = pickle.load(uploaded_file_tab1)
                st.session_state['results'] = loaded_data
                # 현재 처리한 파일의 ID를 session_state에 기록합니다.
                st.session_state.last_uploaded_file_id = current_file_id
                st.success(f"'{uploaded_file_tab1.name}' 파일의 상세 결과를 성공적으로 불러왔습니다.")
                # 결과를 즉시 반영하고, 불필요한 재실행을 막기 위해 st.rerun()을 호출합니다.
                st.rerun() 
            except Exception as e:
                st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
    st.divider()

    # --- 결과 표시 로직 (기존 로직을 session_state 확인 후 실행하도록 변경) ---
    # session_state에 결과가 있을 경우 (새로 실행했거나, 불러왔거나)
    if 'results' in st.session_state and st.session_state['results']:
        results = st.session_state['results']
        
        # 불러온 결과의 이름 표시
        st.subheader(f"📑 결과 요약: {results.get('name', '신규 백테스트')}")

        # --- 아래는 기존의 결과 표시 코드와 거의 동일합니다 ---
        prices = results['prices']
        failed_tickers = results['failed_tickers']
        # [수정] 예전 .pkl 파일과 호환되도록 수정
        culprit_tickers = results.get('culprit_tickers', [results.get('culprit_ticker')])
        config = results['config']; currency_symbol = results['currency_symbol']; etf_df = results['etf_df']
        
        timeseries = results['timeseries']
        cumulative_returns = timeseries['portfolio_value']
        benchmark_cumulative = timeseries['benchmark_value']
        strategy_growth = timeseries['strategy_growth']
        benchmark_growth = timeseries['benchmark_growth']
        
        investment_mode = results['investment_mode']; target_weights = results['target_weights']; initial_cap = results['initial_cap']
        metrics = results['metrics']; portfolio_returns = results['portfolio_returns']; benchmark_returns = results['benchmark_returns']

        with st.expander("1. 백테스트 설정 확인"):
            display_config = config.copy()
            # JSON으로 변환할 수 없는 객체들을 문자열로 변환
            if isinstance(display_config.get('start_date'), datetime):
                display_config['start_date'] = display_config['start_date'].strftime('%Y-%m-%d')
            if isinstance(display_config.get('end_date'), datetime):
                display_config['end_date'] = display_config['end_date'].strftime('%Y-%m-%d')
            display_config.pop('tickers', None)
            st.json(display_config)
        

        st.header("2. 데이터 로딩 정보")
        actual_start_date_str = prices.index[0].strftime('%Y-%m-%d')
        requested_start_date_str = pd.to_datetime(config['start_date']).strftime('%Y-%m-%d')

        # culprit_ticker가 이제 culprit_tickers (리스트)로 변경되었습니다.
        if culprit_tickers:
            culprit_names = []
            for ticker in culprit_tickers:
                name = ticker
                if etf_df is not None:
                    match = etf_df[etf_df['Ticker'] == ticker]
                    if not match.empty:
                        name = match.iloc[0]['Name']
                culprit_names.append(f"'{name}'({ticker})")

            # 원인 제공자가 하나일 때와 여러 개일 때 메시지를 다르게 구성
            if len(culprit_tickers) == 1:
                culprits_str = culprit_names[0]
                reason_str = "의 데이터가 가장 늦게 시작되어"
            else:
                culprits_str = ', '.join(culprit_names)
                reason_str = " 등의 데이터가 가장 늦게 시작되어"
            
            st.warning(f"⚠️ {culprits_str} {reason_str}, 모든 자산이 존재하는 **{actual_start_date_str}**부터 백테스트를 시작합니다.")
        
        # 특정 원인 제공자는 없지만, 주말/휴일 등의 이유로 시작일이 변경된 경우
        elif actual_start_date_str > requested_start_date_str:
            st.info(f"💡 요청하신 기간의 첫 거래일인 **{actual_start_date_str}**부터 백테스트를 시작합니다.")

        if failed_tickers: 
            st.warning(f"다운로드에 실패한 티커가 있습니다: {', '.join(failed_tickers)}")
        
        st.subheader("데이터 미리보기 (최근 5일)")
        display_df = prices.tail().copy()
        new_column_names = []
        for ticker in display_df.columns:
            full_name = ticker
            if etf_df is not None:
                match = etf_df[etf_df['Ticker'] == ticker]
                if not match.empty: full_name = match.iloc[0]['Name']
            new_column_names.append(full_name)
        display_df.columns = new_column_names
        st.dataframe(display_df.style.format("{:,.0f}"))

        st.header("3. 백테스트 결과")
        
        # --- 👇 [교체] 카나리아 모멘텀 vs 벤치마크 가격 비교 그래프 (백테스트 기준 적용) ---
        st.subheader(" 카나리아 모멘텀 추이 vs. 벤치마크 가격")
        
        # 1. 필요한 데이터 가져오기
        prices = results.get('prices')
        config = results.get('config')
        
        if prices is None or config is None:
            st.warning("그래프를 그리는데 필요한 데이터(가격, 설정)가 결과에 포함되지 않았습니다.")
        else:
            # 2. 그래프용 전체 기간 모멘텀 계산 (헬퍼 함수 사용)
            full_momentum_scores = calculate_full_momentum(prices, config)
        
            # 3. 사용자의 '백테스트 기준'과 '리밸런싱 기준일'에 따라 데이터 가공
            backtest_type = config.get('backtest_type', '일별')
            rebalance_day = config.get('rebalance_day', '월말') # '월초'/'월말' 설정 가져오기
        
            if backtest_type == '월별':
                if rebalance_day == '월초':
                    # 월초 기준: 월 시작(Month Start)의 첫번째 데이터로 리샘플링
                    display_momentum = full_momentum_scores.resample('MS').first()
                    display_prices = prices.resample('MS').first()
                    #st.caption("월별 백테스트 기준: '월초' 설정이 적용되어 표시됩니다.")
                else: # '월말'
                    # 월말 기준: 월 끝(Month End)의 마지막 데이터로 리샘플링
                    display_momentum = full_momentum_scores.resample('M').last()
                    display_prices = prices.resample('M').last()
                    #st.caption("월별 백테스트 기준: '월말' 설정이 적용되어 표시됩니다.")
            else: # '일별'
                display_momentum = full_momentum_scores
                display_prices = prices
                #st.caption("일별 백테스트 기준: 일별 데이터로 표시됩니다.")
        
            # 4. 표시할 데이터 시리즈 추출
            canary_tickers = config['tickers']['CANARY']
            benchmark_ticker = config['benchmark']
        
            if canary_tickers and benchmark_ticker in display_prices.columns:
                canary_momentum = display_momentum[canary_tickers].mean(axis=1)
                benchmark_price = display_prices[benchmark_ticker]
        
                # 5. 이중 축 그래프 그리기 (이하 동일)
                fig_mom, ax_mom = plt.subplots(figsize=(10, 5))
                ax_price = ax_mom.twinx()
        
                # 왼쪽 축: 카나리아 모멘텀
                ax_mom.plot(canary_momentum.index, canary_momentum, 
                            label=f'Canary Momentum ({",".join(canary_tickers)})', 
                            color='blue', linewidth=1.0)
                ax_mom.set_ylabel('카나리아 모멘텀 점수', fontsize=12)
                ax_mom.tick_params(axis='y')
        
                # 오른쪽 축: 벤치마크 가격
                ax_price.plot(benchmark_price.index, benchmark_price, 
                              label=f'Benchmark Price ({benchmark_ticker})', 
                              color='grey', linewidth=1.0)
                ax_price.set_ylabel(f'{benchmark_ticker} 가격', fontsize=12)
                ax_price.tick_params(axis='y')

                # --- [추가] 카나리아 모멘텀이 0 이상인 구간에 배경 음영 추가 ---
                # 1. 모멘텀이 0 이상인 구간을 True, 아니면 False로 표시
                is_positive = canary_momentum >= 0
                # 2. True인 구간들의 시작과 끝을 찾아 axvspan으로 배경색을 칠함
                start_date = None
                for i in range(len(is_positive)):
                    # 현재 시점에 0 이상이고, 이전 시점에는 0 미만이었거나 첫 시작이면 -> 상승 구간 시작
                    if is_positive[i] and (i == 0 or not is_positive[i-1]):
                        start_date = canary_momentum.index[i]
                    # 현재 시점에 0 미만이고, 이전 시점에 0 이상이었으면 -> 상승 구간 끝
                    elif not is_positive[i] and (i > 0 and is_positive[i-1]) and start_date:
                        end_date = canary_momentum.index[i]
                        ax_mom.axvspan(start_date, end_date, facecolor='lightgreen', alpha=0.3)
                        start_date = None
                # 마지막까지 상승 구간이 이어졌을 경우 처리
                if start_date:
                    ax_mom.axvspan(start_date, canary_momentum.index[-1], facecolor='lightgreen', alpha=0.3)
                # --- 추가 로직 끝 ---   
        
                ax_mom.axhline(0, color='red', linestyle=':', linewidth=1.0)
                ax_mom.set_title('카나리아 모멘텀 vs. 벤치마크 가격', fontsize=16)
                ax_mom.set_xlabel('날짜', fontsize=12)
                ax_mom.grid(True, which="both", ls="--", linewidth=0.5)
        
                lines, labels = ax_mom.get_legend_handles_labels()
                lines2, labels2 = ax_price.get_legend_handles_labels()
                ax_mom.legend(lines + lines2, labels + labels2, loc='upper left')
                
                st.pyplot(fig_mom)
            else:
                st.warning("카나리아 또는 벤치마크 자산 데이터를 찾을 수 없습니다.")
             
        
        if config['monthly_contribution'] > 0:
            with st.expander("💡 적립식 투자 결과, 어떻게 해석해야 할까요? (클릭하여 보기)"):
                st.markdown("""
                | 항목 | 변경 여부 | 이유 |
                | :--- | :--- | :--- |
                | **최종 자산, 누적/하락폭 그래프** | **변경됨** | 수익과 **추가 원금**이 모두 반영된 **'나의 실제 계좌 잔고'** |
                | **CAGR, MDD, 연/월별 수익률** | **변경되지 않음** | 추가 원금 효과를 제외한 **'순수 투자 전략'**의 성과 |
                """)

        st.subheader("📈 성과 요약")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### **전략 (Strategy)**")

            # --- 백테스트 기간 표시 ---
            start_date_str = results['prices'].index[0].strftime('%Y-%m-%d')
            end_date_str = results['prices'].index[-1].strftime('%Y-%m-%d')
            st.metric("분석 기간", f"{start_date_str} ~ {end_date_str}", help="데이터가 존재하는 실제 분석 기간입니다.")

            # --- [추가] 실행 엔진 설정 요약 표시 ---
            engine_settings_str = (
                f"{config['backtest_type']} | "
                f"{config['rebalance_freq']} 리밸런싱 ({config['rebalance_day']}) | "
                f"거래비용: {config['transaction_cost']:.2%} | "
                f"무위험: {config['risk_free_rate']:.2%}"
            )
            st.markdown(f"<p style='font-size: 0.85em; color: #555; margin-top: -10px;'>{engine_settings_str}</p>", unsafe_allow_html=True)

            # --- 손익 % 계산 ---
            total_profit = metrics['total_profit']
            total_contribution = metrics['total_contribution']
            profit_percentage = (total_profit / total_contribution) if total_contribution != 0 else 0
            profit_delta = f"손익: {currency_symbol}{total_profit:,.0f} ({profit_percentage:.2%})"
            
            st.metric("최종 자산", f"{currency_symbol}{metrics['final_assets']:,.0f}", profit_delta)
            st.metric("총 투자 원금", f"{currency_symbol}{metrics['total_contribution']:,.0f}")
            
            # --- 총 투자 원금 상세 내역 표시 ---
            if config['monthly_contribution'] > 0:
                num_contributions = len(target_weights.index) - 1 if len(target_weights.index) > 0 else 0
                breakdown_str = f"(초기: {currency_symbol}{config['initial_capital']:,.0f} + 추가: {currency_symbol}{config['monthly_contribution']:,.0f} x {num_contributions}회)"
                st.markdown(f"<p style='font-size: 0.8em; color: #555; margin-top: -10px;'>{breakdown_str}</p>", unsafe_allow_html=True)
            
            st.metric("CAGR (연평균 수익률)", f"{metrics['cagr']:.2%}", help="현금흐름(추가입금)을 제외한 순수 전략의 연평균 복리수익률입니다.")
            
            mdd_help = f"기간: {metrics['mdd_start'].strftime('%Y-%m-%d')} ~ {metrics['mdd_end'].strftime('%Y-%m-%d')}"
            st.metric("MDD (최대 낙폭)", f"{metrics['mdd']:.2%}", help=mdd_help)
            
            volatility_help = "수익률의 변동폭을 나타내는 지표로, 수치가 높을수록 가격 변동 위험이 크다는 의미입니다. 연율화된 수익률의 표준편차로 계산됩니다."
            sharpe_help = "무위험 자산 대비 초과 수익률을 변동성으로 나눈 값으로, 위험 대비 수익성을 나타냅니다. 수치가 높을수록 감수한 위험 대비 높은 수익을 얻었다는 의미입니다."
            
            st.metric("Volatility (변동성)", f"{metrics['volatility']:.2%}", help=volatility_help)
            st.metric("Sharpe Ratio (샤프 지수)", f"{metrics['sharpe_ratio']:.2f}", help=sharpe_help)
            st.metric("Win Rate (승률)", f"{metrics['win_rate']:.2%}", help="전체 투자 기간(일/월) 중 수익을 낸 기간의 비율입니다.")            

        with col2:
            st.markdown(f"##### **벤치마크 ({config['benchmark']})**")

            # --- [수정] 백테스트 기간 표시 (help 제거) ---
            start_date_str = results['prices'].index[0].strftime('%Y-%m-%d')
            end_date_str = results['prices'].index[-1].strftime('%Y-%m-%d')
            st.metric("분석 기간", f"{start_date_str} ~ {end_date_str}")
            
            st.markdown(f"<p style='font-size: 0.85em; color: transparent; margin-top: -10px;'>&nbsp;</p>", unsafe_allow_html=True)
            
            # --- 벤치마크 손익 % 계산 ---
            bm_total_profit = metrics['bm_total_profit']
            bm_total_contribution = metrics['bm_total_contribution']
            bm_profit_percentage = (bm_total_profit / bm_total_contribution) if bm_total_contribution != 0 else 0
            bm_profit_delta = f"손익: {currency_symbol}{bm_total_profit:,.0f} ({bm_profit_percentage:.2%})"

            st.metric("최종 자산", f"{currency_symbol}{metrics['bm_final_assets']:,.0f}", bm_profit_delta)
            st.metric("총 투자 원금", f"{currency_symbol}{metrics['bm_total_contribution']:,.0f}")

            # --- 총 투자 원금 상세 내역 표시 (벤치마크) ---
            if config['monthly_contribution'] > 0:
                num_contributions = len(target_weights.index) - 1 if len(target_weights.index) > 0 else 0
                breakdown_str = f"(초기: {currency_symbol}{config['initial_capital']:,.0f} + 추가: {currency_symbol}{config['monthly_contribution']:,.0f} x {num_contributions}회)"
                st.markdown(f"<p style='font-size: 0.8em; color: #555; margin-top: -10px;'>{breakdown_str}</p>", unsafe_allow_html=True)

            st.metric("CAGR (연평균 수익률)", f"{metrics['bm_cagr']:.2%}")

            bm_mdd_help = f"{metrics['bm_mdd_start'].strftime('%Y-%m-%d')} ~ {metrics['bm_mdd_end'].strftime('%Y-%m-%d')}"
            st.metric("MDD (최대 낙폭)", f"{metrics['bm_mdd']:.2%}", help=bm_mdd_help)

            st.metric("Volatility (변동성)", f"{metrics['bm_volatility']:.2%}")
            st.metric("Sharpe Ratio (샤프 지수)", f"{metrics['bm_sharpe_ratio']:.2f}")
            st.metric("Win Rate (승률)", f"{metrics['bm_win_rate']:.2%}")
        
        st.subheader("📊 누적 수익 그래프")
        fig, ax = plt.subplots(figsize=(10, 5))
        if not investment_mode.empty:
            mode_changes = investment_mode.loc[investment_mode.shift(1) != investment_mode].index.tolist()
            if investment_mode.index[0] not in mode_changes: mode_changes.insert(0, investment_mode.index[0])
            for i in range(len(mode_changes)):
                start_interval = mode_changes[i]
                end_interval = mode_changes[i+1] if i+1 < len(mode_changes) else cumulative_returns.index[-1]
                mode = investment_mode.loc[start_interval]
                color = 'lightgreen' if mode == 'Aggressive' else 'lightyellow'
                ax.axvspan(start_interval, end_interval, facecolor=color, alpha=0.3)
        line1, = ax.plot(cumulative_returns.index, cumulative_returns, label='Strategy', color='royalblue', linewidth=1.0)
        line2, = ax.plot(benchmark_cumulative.index, benchmark_cumulative, label='Benchmark', color='grey', linewidth=1.0)
        legend_handles = [line1, line2, Patch(facecolor='lightgreen', label='Aggressive'), Patch(facecolor='lightyellow', label='Defensive')]
        ax.set_title('Cumulative Value Over Time', fontsize=16)
        ax.set_xlabel('Date', fontsize=12); ax.set_ylabel('Portfolio Value', fontsize=12)
        formatter = mtick.FuncFormatter(lambda y, _: format_large_number(y, symbol=currency_symbol))
        ax.yaxis.set_major_formatter(formatter)
        ax.legend(handles=legend_handles, loc='upper left', fontsize=10); ax.grid(True, which="both", ls="--", linewidth=0.5)
        st.pyplot(fig)
        
        st.markdown("---")
        st.header("🔬 상세 분석")
        
        st.subheader("📅 연도별 수익률")
        col1_annual, col2_annual = st.columns([1, 2])
        returns_freq = config['backtest_type'].split(' ')[0]
        if returns_freq == '일별':
            monthly_pf_returns_for_annual = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_bm_returns_for_annual = benchmark_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        else:
            monthly_pf_returns_for_annual = portfolio_returns; monthly_bm_returns_for_annual = benchmark_returns
        annual_returns = monthly_pf_returns_for_annual.resample('A').apply(lambda x: (1 + x).prod() - 1).to_frame(name="Strategy")
        bm_annual_returns = monthly_bm_returns_for_annual.resample('A').apply(lambda x: (1 + x).prod() - 1).to_frame(name="Benchmark")
        annual_df = pd.concat([annual_returns, bm_annual_returns], axis=1)
        annual_df.index = annual_df.index.year
        with col1_annual: st.dataframe(annual_df.style.format("{:.2%}"))
        with col2_annual:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            annual_df.plot(kind='bar', ax=ax2, color=['royalblue', 'grey']); ax2.set_title('Annual Returns', fontsize=16)
            ax2.set_xlabel('Year', fontsize=12); ax2.set_ylabel('Return', fontsize=12); ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax2.tick_params(axis='x', rotation=45); ax2.grid(axis='y', linestyle='--', linewidth=0.5); st.pyplot(fig2)

        st.subheader("📉 하락폭(Drawdown) 추이")
        strategy_dd = (strategy_growth / strategy_growth.cummax() - 1)
        benchmark_dd = (benchmark_growth / benchmark_growth.cummax() - 1)
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(strategy_dd.index, strategy_dd, label='Strategy Drawdown', color='royalblue', linewidth=1.0)
        ax3.plot(benchmark_dd.index, benchmark_dd, label='Benchmark Drawdown', color='grey', linewidth=1.0)
        ax3.fill_between(strategy_dd.index, strategy_dd, 0, color='royalblue', alpha=0.1)
        ax3.set_title('Drawdown Over Time', fontsize=16)
        ax3.set_xlabel('Date', fontsize=12); ax3.set_ylabel('Drawdown', fontsize=12); ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax3.legend(loc='lower right', fontsize=10); ax3.grid(True, which="both", ls="--", linewidth=0.5); st.pyplot(fig3)
        
        st.subheader("🗓️ 월별 수익률 히트맵")
        if not monthly_pf_returns_for_annual.empty:
            heatmap_df = monthly_pf_returns_for_annual.to_frame(name='Return').copy()
            heatmap_df['Year'] = heatmap_df.index.year; heatmap_df['Month'] = heatmap_df.index.month
            heatmap_pivot = heatmap_df.pivot_table(index='Year', columns='Month', values='Return', aggfunc='sum')
            heatmap_pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            monthly_avg = heatmap_pivot.mean(); heatmap_pivot.loc['Average'] = monthly_avg
            st.dataframe(heatmap_pivot.style.format("{:.2%}", na_rep="").background_gradient(cmap='RdYlGn', axis=None))

        st.subheader("💎 개별 자산 성과 비교 (Buy & Hold)")
        with st.spinner('개별 자산 성과 계산 중...'):
            all_used_tickers = prices.columns.tolist(); asset_perf_list = []
            asset_returns_all = prices.pct_change()
            for asset in all_used_tickers:
                asset_returns = asset_returns_all[asset].dropna()
                if not asset_returns.empty:
                    asset_cum_returns = (1 + asset_returns).cumprod(); asset_first_date = asset_cum_returns.first_valid_index()
                    asset_years = (asset_cum_returns.index[-1] - asset_first_date).days / 365.25 if asset_first_date is not None else 0
                    asset_cagr, asset_mdd, asset_vol, asset_sharpe = 0, 0, 0, 0
                    if asset_years > 0:
                        asset_cagr = (asset_cum_returns.iloc[-1])**(1/asset_years) - 1
                        asset_mdd, _, _ = get_mdd_details(asset_cum_returns)
                        trading_periods_asset = 252; asset_vol = asset_returns.std() * np.sqrt(trading_periods_asset)
                        asset_sharpe = (asset_cagr - config['risk_free_rate']) / asset_vol if asset_vol != 0 else 0
                    full_name = ""; 
                    if etf_df is not None:
                        match = etf_df[etf_df['Ticker'] == asset]
                        if not match.empty: full_name = match.iloc[0]['Name']
                    asset_perf_list.append({'Ticker': asset, 'Name': full_name, 'CAGR': asset_cagr, 'Volatility': asset_vol, 'MDD': asset_mdd, 'Sharpe Ratio': asset_sharpe})
            if asset_perf_list:
                asset_perf_df = pd.DataFrame(asset_perf_list).set_index(['Ticker', 'Name'])
                st.dataframe(asset_perf_df.style.format({'CAGR': '{:.2%}', 'Volatility': '{:.2%}', 'MDD': '{:.2%}', 'Sharpe Ratio': '{:.2f}'}))
                
        with st.expander("⚖️ 월별 리밸런싱 내역 보기 (최근 12개월)"):
            recent_weights = target_weights[target_weights.index > (target_weights.index.max() - pd.DateOffset(months=12))]
            for date, weights in recent_weights.iterrows():
                holdings = weights[weights > 0]; display_month_str = (date + pd.DateOffset(months=1)).strftime('%Y-%m')
                if not holdings.empty:
                    holding_list = []
                    for ticker, weight in holdings.items():
                        full_name = ticker 
                        if etf_df is not None:
                            match = etf_df[etf_df['Ticker'] == ticker]
                            if not match.empty: full_name = match.iloc[0]['Name']
                        holding_list.append(f"{full_name} ({weight:.0%})")
                    holding_str = ", ".join(holding_list)
                    st.text(f"{display_month_str}: {holding_str}")
                else: st.text(f"{display_month_str}: 현금 (100%)")

        st.markdown("---")
        st.subheader("💾 결과 저장 및 내보내기")
        
        if 'results' in st.session_state and st.session_state['results']:
            col1, col2 = st.columns(2)
        
            with col1:
                st.markdown("##### 1. 현재 세션에 임시 저장")
            
                # (이 부분은 수정 없음)
                default_name = st.session_state['results'].get('name', "나의 모멘텀 전략")
                if 'backtest_save_name' not in st.session_state:
                    st.session_state.backtest_save_name = default_name
            
                st.text_input(
                    "세션에 저장할 이름:",
                    key='backtest_save_name'
                )
                
                if st.button("세션에 저장"):
                    backtest_name_to_save = st.session_state.backtest_save_name
                
                    # ▼▼▼▼▼ 핵심 수정 부분 ▼▼▼▼▼
                    # 현재 결과를 '바로가기'가 아닌 완전한 '복사본'으로 만듭니다.
                    # pickle을 사용하면 저장하는 시점의 데이터를 그대로 스냅샷처럼 찍어낼 수 있습니다.
                    copied_results = pickle.loads(pickle.dumps(st.session_state['results']))
                
                    new_result = {
                        'name': backtest_name_to_save,
                        'data': copied_results  # 원본 대신 생성한 '복사본'을 저장합니다.
                    }
                    st.session_state.saved_results.append(new_result)
                    
                    st.toast(f"✅ '{backtest_name_to_save}' 결과가 세션에 저장되었습니다!", icon="💾")

            

        
            with col2:
                # (파일 다운로드 부분은 수정할 필요 없습니다.)
                st.markdown("##### 2. 파일로 영구 저장")
                st.write(" ") 
                st.write(" ")
                
                result_binary = pickle.dumps(st.session_state['results'])
                file_name_suggestion = st.session_state.get('backtest_save_name', default_name)
        
                st.download_button(
                    label="파일로 다운로드",
                    data=result_binary,
                    file_name=f"{file_name_suggestion}.pkl",
                    mime="application/octet-stream",
                    help="현재 백테스트 결과를 내 컴퓨터에 .pkl 파일로 영구 저장합니다."
                )

                
# --- 2단계: 결과 비교 탭 (업그레이드 버전) ---
with tab2:
    st.header("📊 세션 결과 및 업로드 파일 비교")
    st.divider()

    # --- 파일 업로드 섹션 (수정 없음) ---
    st.subheader("파일에서 결과 불러오기")
    uploaded_files = st.file_uploader(
        "저장된 .pkl 파일을 여기에 업로드하세요.",
        type=['pkl'],
        accept_multiple_files=True,
        key="uploader_tab2"
    )
    if uploaded_files:
        if 'loaded_files' not in st.session_state:
            st.session_state.loaded_files = set()
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.loaded_files:
                try:
                    loaded_data = pickle.load(uploaded_file)
                    new_result = {
                        'name': uploaded_file.name.replace('.pkl', ''),
                        'data': loaded_data
                    }
                    st.session_state.saved_results.append(new_result)
                    st.session_state.loaded_files.add(uploaded_file.name)
                    st.toast(f"✅ '{uploaded_file.name}' 파일을 세션에 추가했습니다!")
                except Exception as e:
                    st.error(f"'{uploaded_file.name}' 파일 처리 중 오류 발생: {e}")
    st.divider()

    # --- 비교 분석 로직 (버튼 방식으로 변경) ---
    # 1. session_state에 필요한 값들을 초기화합니다.
    if 'show_comparison' not in st.session_state:
        st.session_state.show_comparison = False
    if 'last_selected' not in st.session_state:
        st.session_state.last_selected = None

    saved_results_list = st.session_state.saved_results
    
    if not saved_results_list:
        st.info("현재 세션에 저장된 결과가 없습니다.")
    else:
        result_names = [result['name'] for result in saved_results_list]
        
        selected_names = st.multiselect(
            "저장된 결과 목록에서 비교할 항목을 선택하세요.",
            options=result_names
        )

        # 2. 선택 항목이 변경되면, 이전 분석 결과를 숨기도록 상태를 초기화합니다.
        if selected_names != st.session_state.last_selected:
            st.session_state.show_comparison = False
            st.session_state.last_selected = selected_names

        # 3. 체크박스 대신 버튼을 사용합니다.
        if st.button("🚀 비교 분석하기"):
            if selected_names:
                st.session_state.show_comparison = True
                # st.rerun()을 호출하여 버튼 클릭 즉시 결과가 표시되도록 합니다.
                st.rerun()
            else:
                st.warning("비교할 항목을 먼저 선택해주세요.")

        # 4. 버튼 클릭 신호가 True일 때만 분석 결과를 표시합니다.
        if st.session_state.show_comparison and selected_names:
            
            selected_results_structured = [
                result for result in saved_results_list if result['name'] in selected_names
            ]
            
            st.divider()
            st.subheader("📈 성과 요약 비교")
            
            # (이하 모든 테이블 및 그래프 생성 코드는 이전과 동일하게 작동합니다)
            comparison_data = []
            for result_item in selected_results_structured:
                result_name = result_item['name']
                result_data = result_item['data']
                metrics = result_data.get('metrics', {})
                currency = result_data.get('currency_symbol', '$')
                total_profit = metrics.get('total_profit', 0)
                total_contribution = metrics.get('total_contribution', 1)
                final_return_rate = (total_profit / total_contribution) if total_contribution != 0 else 0
                comparison_data.append({
                    "이름": result_name,
                    "최종 자산": f"{currency}{metrics.get('final_assets', 0):,.0f}",
                    "CAGR": metrics.get('cagr', 0),
                    "MDD": metrics.get('mdd', 0),
                    "변동성": metrics.get('volatility', 0),
                    "샤프 지수": metrics.get('sharpe_ratio', 0),
                    "총 투자 원금": f"{currency}{total_contribution:,.0f}",
                    "총 손익": f"{currency}{total_profit:,.0f}",
                    "최종 수익률": final_return_rate  # <-- 최종 수익률 항목 추가

                })
            
            if comparison_data:
                comp_df = pd.DataFrame(comparison_data).set_index("이름")
                st.dataframe(comp_df.style.format({
                    "CAGR": "{:.2%}", "MDD": "{:.2%}", "변동성": "{:.2%}",
                    "샤프 지수": "{:.2f}", "최종 수익률": "{:.2%}"
                }))

            st.divider()
            st.subheader("📊 누적 수익률 비교 그래프")
            
            fig1, ax1 = plt.subplots(figsize=(10, 5))

            for result_item in selected_results_structured:
                result_name = result_item['name']
                result_data = result_item['data']
                
                timeseries = result_data.get('timeseries', {})
                config = result_data.get('config', {})
                portfolio_value = timeseries.get('portfolio_value')
                
                if portfolio_value is not None and not portfolio_value.empty:
                    # 적립식 투자를 고려한 누적 수익률(%)을 계산하는 로직
                    initial_capital = config.get('initial_capital', 0)
                    monthly_contribution = config.get('monthly_contribution', 0)
                    target_weights = result_data.get('target_weights', pd.DataFrame())
                    contribution_dates = target_weights.index

                    monthly_adds = pd.Series(monthly_contribution, index=contribution_dates)
                    monthly_adds = monthly_adds.reindex(portfolio_value.index).fillna(0)
                    
                    if not monthly_adds.empty:
                        # 첫 날 투자 원금은 초기 투자금 + 첫 월 추가 투자금
                        monthly_adds.iloc[0] = initial_capital + monthly_adds.iloc[0]
                    
                    cumulative_contributions = monthly_adds.cumsum()

                    # 수익률(%) = (현재 자산 - 누적 원금) / 누적 원금
                    cumulative_return_pct = ((portfolio_value - cumulative_contributions) / cumulative_contributions.replace(0, np.nan)) * 100
                    
                    ax1.plot(cumulative_return_pct, label=result_name, linewidth=1.0)

            ax1.set_title('Cumulative Return Comparison', fontsize=16)
            ax1.set_xlabel('Date'); ax1.set_ylabel('Cumulative Return (%)')
            ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y:,.0f}%'))
            ax1.legend(loc='upper left'); ax1.grid(True, which="both", ls="--", linewidth=0.5)
            st.pyplot(fig1)

            st.divider()
            st.subheader("📉 하락폭(Drawdown) 비교 그래프")
            
            fig2, ax2 = plt.subplots(figsize=(10, 5))

            for result_item in selected_results_structured:
                result_name = result_item['name']
                result_data = result_item['data']

                timeseries = result_data.get('timeseries', {})
                dd_series = timeseries.get('strategy_drawdown')

                if dd_series is not None:
                    ax2.plot(dd_series, label=result_name, linewidth=1.0)
                    ax2.fill_between(dd_series.index, dd_series, 0, alpha=0.1) # 하락폭 영역 음영 처리

            ax2.set_title('Drawdown Comparison', fontsize=16)
            ax2.set_xlabel('Date'); ax2.set_ylabel('Drawdown')
            ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax2.legend(loc='lower left'); ax2.grid(True, which="both", ls="--", linewidth=0.5)
            st.pyplot(fig2)


            
# --- 페이지 최상단/최하단 이동 버튼 추가 ---
st.markdown("""
    <style>
        .scroll-buttons {
            position: fixed;
            bottom: 70px;
            right: 20px;
            z-index: 1000;
            display: flex;
            flex-direction: column;
        }
        .scroll-buttons a {
            text-decoration: none;
        }
        .scroll-buttons button {
            /* --- 크기 및 모양 수정 --- */
            width: 40px;
            height: 40px;
            border-radius: 50%; /* 원형으로 변경 */
            font-size: 20px; /* 아이콘 크기 */
            padding: 0;
            
            /* --- 기존 스타일 --- */
            background-color: rgba(79, 139, 249, 0.8); /* 약간 투명하게 */
            color: white;
            border: none;
            text-align: center;
            cursor: pointer;
            margin: 4px 0;
            box-shadow: 0 2px 4px 0 rgba(0,0,0,0.2);
            
            /* 아이콘 중앙 정렬 */
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .scroll-buttons button:hover {
            background-color: #3a75d1;
        }
    </style>
    <a id='bottom'></a>
    <div class="scroll-buttons">
        <a href="#top"><button>🔼</button></a>
        <a href="#bottom"><button>🔽</button></a>
    </div>
""", unsafe_allow_html=True)

# =============================================================================
#                  페이지 전체에 워터마크 추가 (맨 마지막에 위치)
# =============================================================================
st.markdown(
    """
    <style>
    .watermark {
        position: fixed; /* 화면에 고정 */
        top: 60px;    /* 하단에서 10px 떨어진 위치 */
        right: 35px;     /* 우측에서 10px 떨어진 위치 */
        opacity: 0.5;    /* 투명도 50% */
        font-size: 12px; /* 글자 크기 */
        color: gray;     /* 글자 색상 */
        z-index: 999;    /* 다른 요소들 위에 표시 */
        pointer-events: none; /* 워터마크가 클릭되지 않도록 설정 */
    }
    </style>
    <div class="watermark">Dev.HJPark</div>
    """,
    unsafe_allow_html=True

)








