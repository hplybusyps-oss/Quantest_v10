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
import plotly.express as px
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



# -----------------------------------------------------------------------------
# 1. GUI 화면 구성 (Streamlit)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="[Quantest] 퀀트 백테스트", page_icon="📈", layout="wide")

# --- [추가] 새로고침 후 메시지를 표시하는 로직 ---
if 'toast_message' in st.session_state:
    # session_state에 저장된 메시지를 toast로 표시 (아이콘은 메시지 종류에 따라 다름)
    st.toast(st.session_state.toast_message, icon="✅")
    del st.session_state.toast_message

@st.cache_data
def load_Stock_list():
    try:
        if getattr(sys, 'frozen', False):
            application_path = os.path.dirname(sys.executable)
        else:
            application_path = os.path.dirname(os.path.abspath(__file__))
        
        csv_path = os.path.join(application_path, 'Stock_list.csv')

        df = pd.read_csv(csv_path, encoding='utf-8')
        
        df['display'] = df['Ticker'] + ' - ' + df['Name']
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=['Ticker', 'Name', 'display'])
    except Exception as e:
        st.error(f"Stock_list.csv 파일을 읽는 중 오류 발생: {e}")
        return None

# --- [추가] .pkl 파일 로드 시 사이드바 상태를 업데이트하는 로직 ---
# st.rerun() 후 스크립트가 다시 시작될 때 이 부분이 먼저 실행됩니다.
if 'config_to_load' in st.session_state:
    loaded_config = st.session_state.config_to_load

    # ── 1. 티커 목록 (자산군) ────────────────────────────────────────────
    etf_df_for_update = load_Stock_list()
    if etf_df_for_update is not None:
        full_display_list = etf_df_for_update['display'].tolist()

        if 'tickers' in loaded_config:
            loaded_tickers = loaded_config['tickers']
            st.session_state.selected_canary     = [item for item in full_display_list if item.split(' - ')[0] in loaded_tickers.get('CANARY', [])]
            st.session_state.selected_aggressive = [item for item in full_display_list if item.split(' - ')[0] in loaded_tickers.get('AGGRESSIVE', [])]
            st.session_state.selected_defensive  = [item for item in full_display_list if item.split(' - ')[0] in loaded_tickers.get('DEFENSIVE', [])]

        if 'benchmark' in loaded_config:
            match = etf_df_for_update[etf_df_for_update['Ticker'] == loaded_config['benchmark']]
            if not match.empty:
                st.session_state.sidebar_benchmark_display = match.iloc[0]['display']

    # ── 2. 기본 설정 ─────────────────────────────────────────────────────
    if 'start_date' in loaded_config:
        st.session_state['sb_start_date'] = pd.to_datetime(loaded_config['start_date']).date()
    if 'end_date' in loaded_config:
        st.session_state['sb_end_date'] = pd.to_datetime(loaded_config['end_date']).date()
    if 'initial_capital' in loaded_config:
        st.session_state['sb_initial_capital'] = int(loaded_config['initial_capital'])
    if 'monthly_contribution' in loaded_config:
        st.session_state['sb_monthly_contribution'] = int(loaded_config['monthly_contribution'])

    # ── 3. 전략 선택 ─────────────────────────────────────────────────────
    if 'strategy_mode' in loaded_config:
        st.session_state['sb_strategy_mode'] = loaded_config['strategy_mode']

    # ── 4. 실행 엔진 설정 ────────────────────────────────────────────────
    if 'backtest_type' in loaded_config:
        st.session_state['sb_backtest_type'] = loaded_config['backtest_type']
    if 'rebalance_freq' in loaded_config:
        st.session_state['sb_rebalance_freq'] = loaded_config['rebalance_freq']
    if 'rebalance_day' in loaded_config:
        st.session_state['sb_rebalance_day'] = loaded_config['rebalance_day']
    if 'transaction_cost' in loaded_config:
        # config에는 소수로 저장 (0.001), 슬라이더는 % 단위 (0.1)
        st.session_state['sb_transaction_cost'] = float(loaded_config['transaction_cost']) * 100
    if 'risk_free_rate' in loaded_config:
        st.session_state['sb_risk_free_rate'] = float(loaded_config['risk_free_rate']) * 100

    # ── 5. 시그널 설정 ───────────────────────────────────────────────────
    mom_params = loaded_config.get('momentum_params', {})
    if mom_params.get('type'):
        st.session_state['sb_momentum_type'] = mom_params['type']
    if mom_params.get('periods'):
        st.session_state['sb_momentum_periods_str'] = ', '.join(map(str, mom_params['periods']))

    # ── 6. 포트폴리오 구성 전략 ──────────────────────────────────────────
    port_params = loaded_config.get('portfolio_params', {})
    if 'use_canary' in port_params:
        st.session_state['sb_use_canary'] = bool(port_params['use_canary'])
    if 'use_hybrid_protection' in port_params:
        st.session_state['sb_use_hybrid_protection'] = bool(port_params['use_hybrid_protection'])
    if 'top_n_aggressive' in port_params:
        st.session_state['sb_top_n_aggressive'] = int(port_params['top_n_aggressive'])
    if 'top_n_defensive' in port_params:
        st.session_state['sb_top_n_defensive'] = int(port_params['top_n_defensive'])

    # ── 7. A-Core 전용 설정 ──────────────────────────────────────────────
    ac = loaded_config.get('acore_config') or {}
    if ac:
        if 'sp500_ticker' in ac:
            st.session_state['sb_acore_sp500_ticker'] = ac['sp500_ticker']
        if 'ma_period' in ac:
            st.session_state['sb_acore_ma_period'] = int(ac['ma_period'])
        if 'vol_window' in ac:
            st.session_state['sb_acore_vol_window'] = int(ac['vol_window'])
        if 'category_cap' in ac:
            st.session_state['sb_acore_category_cap'] = int(ac['category_cap'])
        if 'zscore_threshold' in ac:
            st.session_state['sb_acore_zscore_threshold'] = float(ac['zscore_threshold'])
        if 'category_map' in ac:
            # 대체 자산 레이블 목록을 별도 저장 (위젯 렌더 시 사용)
            st.session_state['sb_acore_alt_tickers'] = [tk for tk, cat in ac['category_map'].items() if cat == '대체']

    # 한 번 사용한 임시 변수는 즉시 삭제
    del st.session_state.config_to_load

etf_df = load_Stock_list()

st.sidebar.title("⚙️ 백테스트 설정")
st.sidebar.header("1. 기본 설정")

start_date = st.sidebar.date_input(
    "시작일",
    value=st.session_state.get('sb_start_date', pd.to_datetime('2007-01-01').date()),
    min_value=date(1970, 1, 1),
    max_value=date.today(),
    key='sb_start_date'
)
end_date = st.sidebar.date_input(
    "종료일",
    value=st.session_state.get('sb_end_date', date.today()),
    min_value=date(1970, 1, 1),
    max_value=date.today(),
    key='sb_end_date'
)

# --- 통화 선택 UI를 제거하고, 나중에 티커 기반으로 자동 결정 ---

initial_capital = st.sidebar.number_input(
    "초기 투자금액",
    value=st.session_state.get('sb_initial_capital', 10000),
    min_value=0,
    step=1000,
    key='sb_initial_capital',
    help="백테스트를 시작하는 초기 총 자산입니다. 통화는 선택된 자산군에 따라 자동 결정됩니다."
)
# 입력된 금액을 천 단위 쉼표로 포맷하여 바로 아래에 표시
st.sidebar.markdown(f"<p style='text-align: right; color: #555; margin-top: -10px; margin-bottom: 10px;'>{initial_capital:,.0f}</p>", unsafe_allow_html=True)


# 월별 추가 투자금액 입력
monthly_contribution = st.sidebar.number_input(
    "월별 추가 투자금액",
    value=st.session_state.get('sb_monthly_contribution', 0),
    min_value=0,
    step=100,
    key='sb_monthly_contribution',
    help="매월 리밸런싱 시점에 추가로 투자할 금액입니다."
)
# 입력된 금액을 천 단위 쉼표로 포맷하여 바로 아래에 표시
st.sidebar.markdown(f"<p style='text-align: right; color: #555; margin-top: -10px;'>{monthly_contribution:,.0f}</p>", unsafe_allow_html=True)


if etf_df is not None:
    benchmark_options = etf_df['display'].tolist()
    
    # --- [수정] 벤치마크 위젯을 session_state와 연동 ---
    # 1. session_state에 저장된 값이 있으면 그것을 기본값으로 사용하고, 없으면 'SPY'를 찾습니다.
    default_benchmark_display = st.session_state.get(
        'sidebar_benchmark_display', 
        next((opt for opt in benchmark_options if 'SPY' in opt), benchmark_options[0])
    )
    
    # 2. 기본값의 인덱스를 찾습니다. (값이 목록에 없을 경우를 대비하여 예외 처리)
    try:
        default_index = benchmark_options.index(default_benchmark_display)
    except ValueError:
        default_index = 0 # 목록에 없으면 첫 번째 항목을 기본값으로 사용
    
    # 3. selectbox에 key와 동적 index를 할당합니다.
    st.sidebar.selectbox(
        "벤치마크 선택",
        options=benchmark_options,
        index=default_index,
        key='sidebar_benchmark_display', # key를 통해 session_state와 연결
        help="전략의 성과를 비교할 기준 지수(벤치마크)를 선택하세요."
    )
    
    # 4. 최종 선택된 값은 항상 session_state에서 가져옵니다.
    benchmark_ticker = st.session_state.sidebar_benchmark_display.split(' - ')[0]
    # --- 수정 끝 ---
else:
    # Stock_list.csv 파일이 없는 경우, 기존의 텍스트 입력 방식을 유지합니다.
    benchmark_ticker = st.sidebar.text_input(
        "벤치마크 티커",
        value='SPY',
        help="전략의 성과를 비교하기 위한 기준 지수(벤치마크의 티커를 입력하세요."
)

st.sidebar.header("2. 전략 선택")
_strategy_options = ('HAA', 'A-Core')
strategy_mode = st.sidebar.radio(
    "운용 전략",
    _strategy_options,
    index=_strategy_options.index(st.session_state.get('sb_strategy_mode', 'HAA')),
    key='sb_strategy_mode',
    help="""
    - **HAA**: 카나리아 자산의 양/음수 모멘텀으로 Risk-On/Off를 결정하는 기존 전략입니다.
    - **A-Core**: 시장 국면(강세/중립/약세)을 3단계로 세분화하고, 샤프비율 기반 랭킹 + 카테고리 캡으로 자산을 선정하는 고도화 전략입니다.
    """
)

st.sidebar.header("3. 실행 엔진 설정")
_bt_options = ('일별', '월별')
backtest_type = st.sidebar.radio(
    "백테스트 데이터 기준",
    _bt_options,
    index=_bt_options.index(st.session_state.get('sb_backtest_type', '월별')),
    key='sb_backtest_type',
    help="""
    백테스트의 시간 단위를 결정합니다.
    - **일별**: 일별 데이터 사용
    - **월별**: 월별 데이터 사용
    """
)
_rf_options = ('월별', '분기별')
rebalance_freq = st.sidebar.radio(
    "리밸런싱 주기",
    _rf_options,
    index=_rf_options.index(st.session_state.get('sb_rebalance_freq', '월별')),
    key='sb_rebalance_freq',
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
_rd_options = ('월말', '월초')
rebalance_day = st.sidebar.radio(
    "리밸런싱 기준일",
    _rd_options,
    index=_rd_options.index(st.session_state.get('sb_rebalance_day', '월말')),
    key='sb_rebalance_day',
    help=rebalance_day_help
)

transaction_cost = st.sidebar.slider(
    "거래 비용 (%)", 0.0, 1.0,
    value=float(st.session_state.get('sb_transaction_cost', 0.1)),
    step=0.01,
    key='sb_transaction_cost',
    help="매수 또는 매도 시 발생하는 **거래 비용(수수료, 슬리피지 등)을 시뮬레이션**합니다. 입력된 값은 편도(one-way) 기준입니다."
)
risk_free_rate = st.sidebar.slider(
    "무위험 수익률 (%)", 0.0, 5.0,
    value=float(st.session_state.get('sb_risk_free_rate', 1.5)),
    step=0.1,
    key='sb_risk_free_rate',
    help="**샤프 지수(Sharpe Ratio) 계산**에 사용되는 무위험 수익률입니다. 일반적으로 미국 단기 국채 금리를 사용하며, 연 수익률 기준으로 입력합니다."
)

# =============================================================================
#           [추가] 사이드바에 '티커 관리' 기능 추가
# =============================================================================
with st.sidebar.expander("티커 관리"):
        # --- [추가] 임시 변경에 대한 안내 메시지 ---
    st.info(
        """
        💡 티커 추가/삭제 임시 저장       
        """
    )
    
    st.markdown("###### 현재 Stock_list.csv 내용")
    
    current_stocks_df = load_Stock_list()
    if current_stocks_df is not None:
        # 높이 제한(height=150)을 해제하여 전체화면 시 화면에 가득 차도록 수정
        st.dataframe(current_stocks_df, use_container_width=True)
    else:
        st.info("Stock_list.csv 파일을 찾을 수 없습니다.")

    # --- [순서 변경] 1. 신규 티커 추가 ---
    st.markdown("---")
    st.markdown("###### 신규 티커 추가")

    with st.form(key='add_ticker_form', clear_on_submit=True):
        new_ticker = st.text_input("추가할 티커 (예: TSLA)").strip().upper()
        # 이름을 선택적으로 입력받도록 칸을 다시 추가합니다.
        new_name_input = st.text_input("종목명 (비워두면 자동으로 검색합니다)").strip()
        
        submitted = st.form_submit_button("티커 추가하기")
        if submitted:
            if new_ticker:
                new_name = new_name_input
                warning_msg = ""
                
                # 사용자가 이름을 비워둔 경우에만 야후 파이낸스 자동 검색 시도
                if not new_name:
                    try:
                        ticker_info = yf.Ticker(new_ticker).info
                        fetched_name = ticker_info.get('longName', ticker_info.get('shortName', ''))
                        if fetched_name:
                            new_name = str(fetched_name)
                        else:
                            new_name = new_ticker
                            warning_msg = "\n(⚠️ 야후 서버에서 이름을 찾지 못해 티커명으로 임시 저장되었습니다. 목록에서 삭제 후 이름을 직접 입력해 추가하실 수 있습니다.)"
                    except Exception:
                        new_name = new_ticker
                        warning_msg = "\n(⚠️ 인터넷/서버 문제로 이름을 찾지 못해 티커명으로 임시 저장되었습니다.)"

                # 2. 파일 경로 설정
                if getattr(sys, 'frozen', False):
                    application_path = os.path.dirname(sys.executable)
                else:
                    application_path = os.path.dirname(os.path.abspath(__file__))
                csv_path = os.path.join(application_path, 'Stock_list.csv')

                # 3. 중복 확인을 위해 기존 파일 읽기
                try:
                    df_from_disk = pd.read_csv(csv_path, encoding='utf-8')
                except FileNotFoundError:
                    df_from_disk = pd.DataFrame(columns=['Ticker'])
                
                # 4. 파일에 저장
                if new_ticker not in df_from_disk['Ticker'].str.upper().values:
                    try:
                        import csv
                        file_exists = os.path.exists(csv_path)
                        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            if not file_exists or os.path.getsize(csv_path) == 0:
                                writer.writerow(['Ticker', 'Name'])
                            writer.writerow([new_ticker, new_name])
                        
                        # 기존 코드에 있던 토스트 알림 기능을 활용해 성공/경고 메시지 전달
                        st.session_state.toast_message = f"'{new_name}' ({new_ticker}) 추가 완료! {warning_msg}"
                        load_Stock_list.clear()
                        
                        st.session_state.temp_selection_agg = st.session_state.get('selected_aggressive', [])
                        st.session_state.temp_selection_def = st.session_state.get('selected_defensive', [])
                        st.session_state.temp_selection_can = st.session_state.get('selected_canary', [])                                           
                        st.rerun()
                    except Exception as e:
                        st.error(f"파일 쓰기 중 오류 발생: {e}")
                else:
                    st.error(f"'{new_ticker}'는 이미 존재하는 티커입니다.")
            else:
                st.warning("추가할 티커를 입력해주세요.")

    # --- [순서 변경] 2. 기존 티커 삭제 ---
    if current_stocks_df is not None and not current_stocks_df.empty:
        st.markdown("---")
        st.markdown("###### 기존 티커 삭제")
        
        tickers_to_delete = st.multiselect(
            "삭제할 티커를 선택하세요.",
            options=current_stocks_df['Ticker'].tolist()
        )
        
        if st.button("티커 삭제하기"):
            if tickers_to_delete:
                try:
                    updated_df = current_stocks_df[~current_stocks_df['Ticker'].isin(tickers_to_delete)]
                    
                    if getattr(sys, 'frozen', False):
                        application_path = os.path.dirname(sys.executable)
                    else:
                        application_path = os.path.dirname(os.path.abspath(__file__))
                    csv_path = os.path.join(application_path, 'Stock_list.csv')

                    updated_df.to_csv(csv_path, index=False, encoding='utf-8')
                    
                    st.success(f"{len(tickers_to_delete)}개의 티커를 삭제했습니다!")                  
                    load_Stock_list.clear()
                    # --- [수정] 값이 없을 경우 에러 없이 빈 리스트를 반환하도록 안전하게 수정 ---
                    st.session_state.temp_selection_agg = st.session_state.get('selected_aggressive', [])
                    st.session_state.temp_selection_def = st.session_state.get('selected_defensive', [])
                    st.session_state.temp_selection_can = st.session_state.get('selected_canary', [])
                    st.rerun()

                except Exception as e:
                    st.error(f"파일 수정 중 오류 발생: {e}")
            else:
                st.warning("삭제할 티커를 먼저 선택해주세요.")

st.sidebar.header("4. 자산군 설정")
if etf_df is not None:
    display_list = etf_df['display'].tolist()

    # --- [수정] session_state 초기화 및 위젯 생성 ---
    # 기본값 목록 정의
    default_canary_list = [d for d in ['TIP - iShares TIPS Bond ETF'] if d in display_list]
    default_aggressive_list = [d for d in ['SPY - SPDR S&P 500 ETF Trust', 'IWM - iShares Russell 2000 ETF', 'EFA - iShares MSCI EAFE ETF', 'VWO - Vanguard FTSE Emerging Markets ETF', 'VNQ - Vanguard Real Estate ETF', 'DBC - Invesco DB Commodity Index Tracking Fund', 'IEF - iShares 7-10 Year Treasury Bond ETF', 'TLT - iShares 20+ Year Treasury Bond ETF'] if d in display_list]
    default_defensive_list = [d for d in ['BIL - SPDR Bloomberg 1-3 Month T-Bill ETF', 'IEF - iShares 7-10 Year Treasury Bond ETF'] if d in display_list]

    # 앱 첫 실행 시에만 기본값으로 session_state를 초기화
    if 'selected_canary' not in st.session_state:
        st.session_state.selected_canary = default_canary_list
    if 'selected_aggressive' not in st.session_state:
        st.session_state.selected_aggressive = default_aggressive_list
    if 'selected_defensive' not in st.session_state:
        st.session_state.selected_defensive = default_defensive_list

    # 위젯은 key를 통해 session_state와 자동으로 동기화됨 (default 인자 불필요)
    with st.sidebar.popover("카나리아 자산 선택하기", use_container_width=True):
        st.multiselect("카나리아 자산 검색", display_list, key='selected_canary')
    with st.sidebar.popover("공격 자산 선택하기", use_container_width=True):
        st.multiselect("공격 자산 검색", display_list, key='selected_aggressive')
    with st.sidebar.popover("방어 자산 선택하기", use_container_width=True):
        st.multiselect("방어 자산 검색", display_list, key='selected_defensive')
    
    # session_state에서 값을 읽어옴
    aggressive_tickers = [s.split(' - ')[0] for s in st.session_state.selected_aggressive]
    defensive_tickers = [s.split(' - ')[0] for s in st.session_state.selected_defensive]
    canary_tickers = [s.split(' - ')[0] for s in st.session_state.selected_canary]

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

st.sidebar.header("5. 시그널 설정")
momentum_type_help = """
- **13612U**: **1, 3, 6, 12개월** 수익률을 평균내어 안정적인 신호를 만듭니다. (HAA 전략 기본값)
- **평균 모멘텀**: 사용자가 **직접 입력한 기간들**의 수익률을 평균냅니다.
- **상대 모멘텀**: 여러 자산 중 특정 기간 동안 가장 많이 상승한 자산을 선택합니다. (상승장 추종에 유리)
"""
_mt_options = ('13612U', '평균 모멘텀', '상대 모멘텀')
momentum_type = st.sidebar.selectbox(
    "모멘텀 종류",
    _mt_options,
    index=_mt_options.index(st.session_state.get('sb_momentum_type', '13612U')),
    key='sb_momentum_type',
    help=momentum_type_help
)
momentum_periods_str = st.sidebar.text_input(
    "모멘텀 기간 (개월, 쉼표로 구분)",
    value=st.session_state.get('sb_momentum_periods_str', '1, 3, 6, 12'),
    key='sb_momentum_periods_str',
    help="""
    - **13612U**: 이 입력값은 **무시**됩니다.
    - **평균 모멘텀**: 사용할 기간을 쉼표로 구분하여 입력합니다. (예: 3, 6, 9)
    - **상대 모멘텀**: 입력된 숫자 중 **첫 번째 값**만 사용합니다. (예: '6' 입력 시 6개월 상대 모멘텀)
    """
)
st.sidebar.header("6. 포트폴리오 구성 전략")
use_canary = st.sidebar.toggle(
    "카나리아 자산 사용 (Risk-On/Off)",
    value=st.session_state.get('sb_use_canary', True),
    key='sb_use_canary',
    help="체크 시, 카나리아 자산의 모멘텀이 양수일 때만 공격 자산에 투자합니다. 해제 시 항상 공격 자산군 내에서만 투자합니다."
)
use_hybrid_protection = st.sidebar.toggle(
    "하이브리드 보호 장치 사용",
    value=st.session_state.get('sb_use_hybrid_protection', True),
    key='sb_use_hybrid_protection',
    help="체크 시, 공격 자산으로 선택되었어도 개별 모멘텀이 음수이면 안전 자산으로 교체합니다."
)
top_n_aggressive = st.sidebar.number_input(
    "공격 자산 Top N",
    min_value=1, max_value=10,
    value=st.session_state.get('sb_top_n_aggressive', 4),
    key='sb_top_n_aggressive',
    help="공격 자산군에서 모멘텀 순위가 높은 상위 N개의 자산을 선택합니다."
)
top_n_defensive = st.sidebar.number_input(
    "방어 자산 Top N",
    min_value=1, max_value=10,
    value=st.session_state.get('sb_top_n_defensive', 1),
    key='sb_top_n_defensive',
    help="방어 자산군에서 모멘텀 순위가 높은 상위 N개의 자산을 선택합니다."
)
weighting_scheme = st.sidebar.selectbox(
    "자산 배분 방식",
    ('동일 비중 (Equal Weight)',),
    help="선택된 자산들에 어떤 비중으로 투자할지 결정합니다. (추후 확장 가능)"
)

# =============================================================================
#    [A-Core 신규] A-Core 전략 전용 사이드바 설정
# =============================================================================
if strategy_mode == 'A-Core':
    st.sidebar.header("7. A-Core 전략 설정")
    
    st.sidebar.markdown("""
    <div style='background-color:#1a3a5c; padding:10px; border-radius:8px; color:#aad4f5; font-size:0.85em; margin-bottom:10px;'>
    ⚡ <b>A-Core 전략 활성화</b><br>
    시장 국면(강세/중립/약세) 3단계 판단 + 샤프비율 랭킹 + 카테고리 캡 적용
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar.expander("🌡️ 시장 국면 판단 설정", expanded=True):
        sp500_ticker_acore = st.text_input(
            "S&P500 지수 티커",
            value=st.session_state.get('sb_acore_sp500_ticker', '^GSPC'),
            key='sb_acore_sp500_ticker',
            help="강세/중립 국면 판단에 사용하는 S&P500 지수 티커입니다. `^GSPC`는 Yahoo Finance 기준 S&P500 지수의 정식 티커이며 오류가 아닙니다. 한국 ETF(예: 069500.KS)로 대체 가능합니다."
        )
        ma_period_acore = st.number_input(
            "이동평균선 기간 (일)",
            min_value=20, max_value=400,
            value=st.session_state.get('sb_acore_ma_period', 200),
            step=10,
            key='sb_acore_ma_period',
            help="S&P500 지수의 이동평균선 기간입니다. 기본값 200일.\n\n[국면 판단 기준]\n- 강세: TIP 모멘텀>0 & S&P500 > MA\n- 중립: TIP 모멘텀>0 & S&P500 < MA\n- 약세: TIP 모멘텀 ≤ 0"
        )
    
    with st.sidebar.expander("📊 랭킹 & 위험관리 설정", expanded=True):
        vol_window_acore = st.number_input(
            "변동성 계산 기간 (일)",
            min_value=20, max_value=120,
            value=st.session_state.get('sb_acore_vol_window', 60),
            step=5,
            key='sb_acore_vol_window',
            help="샤프비율 랭킹의 분모에 사용되는 변동성 계산 기간 (영업일 기준). 기본값 60일(약 3개월)."
        )
        category_cap_acore = st.number_input(
            "카테고리 캡 (카테고리별 최대 편입 수)",
            min_value=1, max_value=5,
            value=st.session_state.get('sb_acore_category_cap', 2),
            step=1,
            key='sb_acore_category_cap',
            help="동일 카테고리(주식/대체/안전)에서 Top N 순위에 들어올 수 있는 최대 자산 수입니다."
        )
        zscore_threshold_acore = st.slider(
            "변동성 Z-Score 임계값",
            min_value=0.5, max_value=3.0,
            value=float(st.session_state.get('sb_acore_zscore_threshold', 1.5)),
            step=0.1,
            key='sb_acore_zscore_threshold',
            help="최근 1개월 변동성이 [12개월 평균 + (표준편차 × 임계값)]을 초과하면 과변동성으로 판단합니다."
        )
    
    with st.sidebar.expander("🗂️ A-Core 자산 카테고리 설정", expanded=True):
        st.caption("대체 자산에 해당하는 종목만 선택해 주세요. 선택되지 않은 나머지 공격 자산은 모두 '주식'으로 자동 분류됩니다.")

        # 선택된 공격 자산 티커 목록
        _agg_tickers_for_cat = [s.split(' - ')[0] for s in st.session_state.get('selected_aggressive', [])]

        if not _agg_tickers_for_cat:
            st.info("공격 자산을 먼저 선택하면 카테고리를 지정할 수 있습니다.")
            acore_category_map = {}
        else:
            # 공격 자산 display 레이블 생성 (티커 → 티커 + 이름)
            _tk_labels = {}
            for tk in _agg_tickers_for_cat:
                label = tk
                if etf_df is not None:
                    m = etf_df[etf_df['Ticker'] == tk]
                    if not m.empty:
                        nm = m.iloc[0]['Name']
                        label = f"{tk} — {nm[:14]}…" if len(nm) > 14 else f"{tk} — {nm}"
                _tk_labels[tk] = label

            _all_labels = list(_tk_labels.values())
            _label_to_tk = {v: k for k, v in _tk_labels.items()}

            # --- [수정된 로직] pkl 파일 로드 시 세션 상태를 직접 업데이트하여 복구 보장 ---
            _DEFAULT_ALT_HINTS = ['GLD', 'IAU', 'SLV', 'DBC', 'USO', 'PDBC', 'GNR', 'VNQ', 'IYR', 'REMX', '411060.KS', '476760.KS', '352560.KS', '305080.KS', '276000.KS', '415920.KS']
            
            _loaded_alt_tickers = st.session_state.get('sb_acore_alt_tickers', None)
            
            # 1. pkl 파일에서 불러온 임시 데이터가 있는 경우
            if _loaded_alt_tickers is not None:
                # 티커를 현재의 라벨 형식으로 변환하여 위젯의 key값('acore_bucket_alt')에 직접 주입
                st.session_state['acore_bucket_alt'] = [_tk_labels[t] for t in _loaded_alt_tickers if t in _tk_labels]
                # 임시 변수 삭제 (다음 렌더링 시 영향을 주지 않도록)
                del st.session_state['sb_acore_alt_tickers']
            
            # 2. 파일 로드가 아니고, 처음 실행 시 기본 힌트를 적용해야 하는 경우
            elif 'acore_bucket_alt' not in st.session_state:
                st.session_state['acore_bucket_alt'] = [_tk_labels[t] for t in _agg_tickers_for_cat if t.upper() in _DEFAULT_ALT_HINTS and t in _tk_labels]

            st.markdown("**대체 자산 선택**")
            # [핵심 수정] default 인자를 제거하고, st.session_state['acore_bucket_alt']가 위젯 상태를 직접 관리하게 함
            chosen_alt_labels = st.multiselect(
                "",
                options=_all_labels,
                key='acore_bucket_alt',
                label_visibility='collapsed'
            )
            
            chosen_alt_tickers = [_label_to_tk[lb] for lb in chosen_alt_labels]

            # 카테고리 맵 생성 (선택된 것은 '대체', 나머지는 모두 '주식')
            acore_category_map = {}
            for tk in _agg_tickers_for_cat:
                if tk in chosen_alt_tickers:
                    acore_category_map[tk] = '대체'
                else:
                    acore_category_map[tk] = '주식'

            # 현재 어떻게 분류되었는지 직관적으로 보여주는 텍스트 안내
            _alt_list = [tk for tk, cat in acore_category_map.items() if cat == '대체']
            _eq_list = [tk for tk, cat in acore_category_map.items() if cat == '주식']
            
            st.markdown("---")
            st.markdown(f"<span style='color:#ff7f0e; font-weight:bold;'>대체 자산:</span> {', '.join(_alt_list) if _alt_list else '없음'}", unsafe_allow_html=True)
            st.markdown(f"<span style='color:#1f77b4; font-weight:bold;'>주식 자산:</span> {', '.join(_eq_list) if _eq_list else '없음'}", unsafe_allow_html=True)
    
    acore_config = {
        'sp500_ticker': sp500_ticker_acore,
        'ma_period': ma_period_acore,
        'vol_window': vol_window_acore,
        'category_cap': category_cap_acore,
        'zscore_threshold': zscore_threshold_acore,
        'category_map': acore_category_map,
    }
else:
    acore_config = None

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
                             'weighting': weighting_scheme},
        'strategy_mode': strategy_mode,
        'acore_config': acore_config,
    }

# 앱이 재실행될 때마다 현재 설정을 가져옴
current_config = gather_current_config()

# 마지막 실행 설정이 있고, 현재 설정과 다를 경우 '변경됨' 플래그를 True로 설정
if 'last_run_config' in st.session_state:
    try:
        settings_are_different = (st.session_state.last_run_config != current_config)
        # dict 비교 결과가 bool이 아닌 경우(예: Series 포함) 안전하게 처리
        if not isinstance(settings_are_different, bool):
            settings_are_different = bool(settings_are_different)
    except Exception:
        settings_are_different = True
    st.session_state.settings_changed = settings_are_different

    # 설정이 변경되었고, 아직 토스트 알림을 보여주지 않았다면
    if settings_are_different and not st.session_state.get('toast_shown', False):
        st.toast("⚙️ 설정이 변경되었습니다!", icon="💡")
        st.session_state.toast_shown = True
else:
    st.session_state.settings_changed = False

# --- 사이드바 설정 변경 감지 로직 끝 ---

# -----------------------------------------------------------------------------
# 2. 백엔드 로직 (데이터 처리 및 백테스트)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_price_data(tickers, start, end, user_start_date):
    try:
        # --- [수정] auto_adjust=False 옵션을 추가합니다 ---
        raw_data = yf.download(
            tickers, 
            start=start, 
            end=pd.to_datetime(end) + pd.DateOffset(days=1),  # yfinance end는 exclusive이므로 +1일
            progress=False,
            auto_adjust=False  # 이 옵션을 추가하면 'Adj Close' 컬럼이 포함됩니다.
        )
        
        if raw_data.empty: 
            st.error("데이터를 다운로드하지 못했습니다."); 
            return None, None, None

        # yfinance MultiIndex 처리: 단일 티커도 MultiIndex를 반환할 수 있음
        if isinstance(raw_data.columns, pd.MultiIndex):
            # 'Adj Close'가 있는지 먼저 확인하고, 없으면 'Close'를 사용하는 로직
            if 'Adj Close' in raw_data.columns.get_level_values(0):
                prices = raw_data['Adj Close'].copy()
            else:
                st.warning("'수정 종가(Adj Close)' 데이터를 일부 티커에서 찾을 수 없어, '종가(Close)'를 기준으로 계산합니다.")
                prices = raw_data['Close'].copy()
        else:
            # 단일 티커이거나 flat 컬럼 구조
            if 'Adj Close' in raw_data.columns:
                prices = raw_data['Adj Close'].copy()
            else:
                st.warning("'수정 종가(Adj Close)' 데이터를 일부 티커에서 찾을 수 없어, '종가(Close)'를 기준으로 계산합니다.")
                prices = raw_data['Close'].copy()
        
        # 단일 티커인 경우 Series → DataFrame으로 변환
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0] if len(tickers) == 1 else 'Price')
        
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
        
        # 사용자가 요청한 진짜 시작일보다 실제 데이터 시작일이 늦은 경우에만 "culprit"으로 간주합니다.
        if actual_latest_start <= pd.to_datetime(user_start_date):
            culprit_tickers = [] # 워밍업 기간에 해당하는 경우는 원인 제공자가 없는 것으로 처리
        
        final_prices = prices[successful_tickers].dropna(axis=0, how='any')

        return final_prices, failed_tickers, culprit_tickers
    except Exception as e:
        st.error(f"데이터 다운로드 중 오류 발생: {e}"); return None, None, None

def calculate_cumulative_returns_with_dca(returns_series, initial_capital, monthly_contribution, contribution_dates):
    """적립식 투자를 반영하여 누적 자산 가치를 계산하는 함수
    
    로직: 매 리밸런싱 기준일에 추가 투자금이 입금된 후, 해당 기간 수익률이 적용됩니다.
    """
    portfolio_values = []
    current_capital = initial_capital

    # 기여금 날짜를 빠르게 조회하기 위해 set으로 변환
    contribution_dates_set = set(contribution_dates)

    for date, ret in returns_series.items():
        # 추가 투자일인 경우, 수익률 적용 전에 추가 투자금 입금
        if date in contribution_dates_set and monthly_contribution > 0:
            current_capital += monthly_contribution
            
        # 수익률에 따라 자산 가치 업데이트
        current_capital *= (1 + ret)
            
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
        ret = prices.pct_change(periods=month * 21)
        # --- [수정된 부분] 더하기 전에 개별 기간에서 발생한 결측치(NaN)를 먼저 0으로 채워줍니다 ---
        ret = ret.fillna(0) 
        returns_dfs.append(ret)
        
    # 모든 기간의 수익률을 합산하여 평균
    if not returns_dfs:
        return pd.DataFrame(0, index=prices.index, columns=prices.columns)
        
    full_momentum_scores = sum(returns_dfs) / len(returns_dfs)
    return full_momentum_scores

# =============================================================================
#   [A-Core 신규] 핵심 백엔드 로직 함수
# =============================================================================

def determine_market_phase(prices: pd.DataFrame, canary_tickers: list, config: dict) -> pd.Series:
    """
    매 리밸런싱 시점마다 시장 국면(강세/중립/약세)을 판단합니다.

    국면 판단 기준:
    - 강세: TIP 평균 모멘텀 > 0  AND  S&P500 > MA
    - 중립: TIP 평균 모멘텀 > 0  AND  S&P500 <= MA
    - 약세: TIP 평균 모멘텀 <= 0  (카나리아 발동)

    Returns:
        pd.Series: 리밸런싱 날짜를 인덱스로 갖는 국면 시리즈 ('강세'/'중립'/'약세')
    """
    acore_cfg = config.get('acore_config', {})
    sp500_ticker = acore_cfg.get('sp500_ticker', '^GSPC')
    ma_period = int(acore_cfg.get('ma_period', 200))

    # 리밸런싱 날짜 추출
    prices_copy = prices.copy()
    day_option = 'last' if config['rebalance_day'] == '월말' else 'first'
    if config['rebalance_freq'] == '분기별':
        prices_copy['_yq'] = prices_copy.index.to_period('Q').strftime('%Y-Q%q')
        rebal_dates = prices_copy.drop_duplicates('_yq', keep=day_option).index
    else:
        prices_copy['_ym'] = prices_copy.index.strftime('%Y-%m')
        rebal_dates = prices_copy.drop_duplicates('_ym', keep=day_option).index

    # S&P500 이동평균선 계산 (사용 가능한 경우)
    sp500_ma = None
    if sp500_ticker in prices.columns:
        sp500_ma = prices[sp500_ticker].rolling(window=ma_period, min_periods=ma_period // 2).mean()

    # 모멘텀 기간 확정 (13612U 기준)
    mom_type = config['momentum_params']['type']
    mom_periods = [1, 3, 6, 12] if mom_type == '13612U' else config['momentum_params'].get('periods', [1, 3, 6, 12])

    market_phase = pd.Series('약세', index=rebal_dates, dtype=str)

    for date in rebal_dates:
        # 1. 카나리아 평균 모멘텀 계산 (TIP 모멘텀)
        valid_canary = [t for t in canary_tickers if t in prices.columns]
        if not valid_canary:
            continue  # 기본값 '약세' 유지

        returns_list = []
        for month in mom_periods:
            past_date = date - pd.DateOffset(months=month)
            if past_date < prices.index[0]:
                returns_list.append(pd.Series(0.0, index=valid_canary))
                continue
            idx = prices.index.get_indexer([past_date], method='nearest')[0]
            past_prices = prices.iloc[idx][valid_canary]
            curr_prices = prices.loc[date, valid_canary]
            # 0으로 나누기 방지
            with np.errstate(divide='ignore', invalid='ignore'):
                ret = np.where(past_prices != 0, curr_prices / past_prices - 1, 0.0)
            returns_list.append(pd.Series(ret, index=valid_canary))

        canary_momentum = sum(returns_list) / len(returns_list)
        avg_canary_score = canary_momentum.mean() if not canary_momentum.empty else -1

        # 2. 약세 국면: 카나리아 모멘텀 <= 0 → 기본값 유지
        if avg_canary_score <= 0:
            continue

        # 3. 강세/중립 국면: S&P500 vs MA 비교
        if sp500_ma is not None and date in sp500_ma.index and not pd.isna(sp500_ma.loc[date]):
            sp500_price_idx = prices.index.get_indexer([date], method='nearest')[0]
            sp500_price = prices.iloc[sp500_price_idx].get(sp500_ticker, None)
            if sp500_price is not None and not pd.isna(sp500_price):
                market_phase[date] = '강세' if sp500_price > sp500_ma.loc[date] else '중립'
            else:
                market_phase[date] = '중립'
        else:
            market_phase[date] = '중립'

    return market_phase


def calculate_acore_momentum(date: pd.Timestamp, prices: pd.DataFrame,
                              config: dict, market_phase: str) -> pd.Series:
    """
    국면에 따라 가중/평균 모멘텀 점수를 계산합니다.

    ■ 평균 모멘텀 (중립/약세 국면)
        score = (r1 + r3 + r6 + r12) / 4
        → 1, 3, 6, 12개월 수익률에 동일 가중치(각 25%) 부여

    ■ 가중 모멘텀 (강세 국면)
        weights = {1개월: 4/10, 3개월: 3/10, 6개월: 2/10, 12개월: 1/10}
        score = r1×0.4 + r3×0.3 + r6×0.2 + r12×0.1
        → 가중치 합 = 1.0 (수익률 자체를 부풀리지 않음)
        → 단기일수록 높은 비중을 줌으로써 최근 추세에 더 빠르게 반응

    Returns:
        pd.Series: 티커별 모멘텀 점수 (수익률 단위, 부풀림 없음)
    """
    tickers = [t for t in prices.columns if t in prices.columns]

    returns_by_month = {}
    mom_periods = [1, 3, 6, 12]
    for month in mom_periods:
        past_date = date - pd.DateOffset(months=month)
        if past_date < prices.index[0]:
            returns_by_month[month] = pd.Series(0.0, index=tickers)
            continue
        idx = prices.index.get_indexer([past_date], method='nearest')[0]
        past_prices = prices.iloc[idx][tickers]
        curr_prices = prices.loc[date, tickers]
        returns_by_month[month] = curr_prices / past_prices - 1

    if market_phase == '강세':
        # 가중 모멘텀: 단기일수록 높은 비중, 가중치 합 = 1.0
        # 1개월(40%) > 3개월(30%) > 6개월(20%) > 12개월(10%)
        score = (
            returns_by_month[1]  * 0.4 +
            returns_by_month[3]  * 0.3 +
            returns_by_month[6]  * 0.2 +
            returns_by_month[12] * 0.1
        )
    else:
        # 평균 모멘텀: 중립/약세 국면, 동일 가중치 각 25%, 합 = 1.0
        score = (
            returns_by_month[1]  * 0.25 +
            returns_by_month[3]  * 0.25 +
            returns_by_month[6]  * 0.25 +
            returns_by_month[12] * 0.25
        )

    return score


def calculate_volatility_zscore_filter(date: pd.Timestamp, prices: pd.DataFrame,
                                        ticker: str, vol_window: int = 90,
                                        zscore_threshold: float = 1.5) -> bool:
    """
    변동성 Z-Score 필터: 비정상적 과변동성 자산을 제외합니다.

    조건: 최근 1개월 변동성 > 12개월 평균 변동성 + (표준편차 × 임계값)
    True 반환 → 정상 (통과), False 반환 → 과변동성 (탈락)
    """
    if ticker not in prices.columns:
        return False

    # 최근 1개월(21영업일) 일별 수익률 변동성
    price_hist = prices[ticker].loc[:date]
    if len(price_hist) < vol_window + 21:
        return True  # 데이터 부족 시 필터 미적용 (통과)

    daily_returns = price_hist.pct_change().dropna()

    # 12개월(252영업일)치의 1개월 롤링 변동성 시계열
    rolling_vol_1m = daily_returns.rolling(21).std() * np.sqrt(252)
    hist_vol_series = rolling_vol_1m.iloc[-252:].dropna()

    if len(hist_vol_series) < 6:
        return True  # 불충분한 히스토리 → 통과

    # 최근 1개월 변동성
    recent_vol = daily_returns.iloc[-21:].std() * np.sqrt(252)

    vol_mean = hist_vol_series.mean()
    vol_std = hist_vol_series.std()

    threshold = vol_mean + vol_std * zscore_threshold
    return recent_vol <= threshold  # 임계 초과 시 False (탈락)


def construct_acore_portfolio(prices: pd.DataFrame, config: dict,
                               successful_tickers: list,
                               market_phase_series: pd.Series) -> tuple:
    """
    A-Core 전략의 포트폴리오를 구성합니다.

    핵심 로직:
    1. 국면별 모멘텀 가중 방식 적용
    2. 절대 모멘텀 필터 (모멘텀 > 0만 매수 대상)
    3. 변동성 Z-Score 필터 (과변동성 자산 제외)
    4. 샤프비율 랭킹 = 모멘텀 / 변동성
    5. 카테고리 캡 적용 후 Top N 선정
    6. 약세 국면: 방어 자산 중 모멘텀 1위 자산에 100% 집중

    Returns:
        (target_weights DataFrame, investment_mode Series, phase_scores_log dict)
    """
    acore_cfg = config.get('acore_config', {})
    vol_window = int(acore_cfg.get('vol_window', 90))
    category_cap = int(acore_cfg.get('category_cap', 2))
    zscore_threshold = float(acore_cfg.get('zscore_threshold', 1.5))
    category_map = acore_cfg.get('category_map', {})

    params = config['portfolio_params']
    top_n = params.get('top_n_aggressive', 4)

    canary_tickers = [t for t in config['tickers']['CANARY'] if t in successful_tickers]
    aggressive_assets = [t for t in config['tickers']['AGGRESSIVE'] if t in successful_tickers]
    defensive_assets = [t for t in config['tickers']['DEFENSIVE'] if t in successful_tickers]

    rebal_dates = market_phase_series.index
    target_weights = pd.DataFrame(0.0, index=rebal_dates, columns=prices.columns)
    investment_mode = pd.Series(index=rebal_dates, dtype=str)
    phase_scores_log = {}

    def get_vol(ticker, date):
        price_hist = prices[ticker].loc[:date]
        if len(price_hist) < vol_window + 1:
            return np.nan
        daily_ret = price_hist.pct_change().dropna().iloc[-vol_window:]
        if len(daily_ret) < 5:
            return np.nan
        return daily_ret.std() * np.sqrt(252)

    def get_category(ticker):
        return category_map.get(ticker.upper(), '기타')

    for date in rebal_dates:
        phase = market_phase_series.get(date, '약세')
        log_details = {}

        # ── 약세 국면 ──────────────────────────────────────────────────
        if phase == '약세':
            investment_mode[date] = '약세 (Defensive)'
            mom_all = calculate_acore_momentum(date, prices, config, '약세')
            def_scores = {}
            for ticker in defensive_assets:
                mom_score = mom_all.get(ticker, np.nan) if hasattr(mom_all, 'get') else (mom_all[ticker] if ticker in mom_all.index else np.nan)
                
                if pd.isna(mom_score) or mom_score <= 0:
                    log_details[ticker] = {'score': mom_score, 'category': '방어', 'status': '모멘텀 음수'}
                    continue
                if not calculate_volatility_zscore_filter(date, prices, ticker, vol_window, zscore_threshold):
                    log_details[ticker] = {'score': mom_score, 'category': '방어', 'status': '과변동성 탈락'}
                    continue
                    
                def_scores[ticker] = mom_score
                log_details[ticker] = {'score': mom_score, 'category': '방어', 'status': '후보'}

            selected_assets = []
            if def_scores:
                best_def = max(def_scores, key=def_scores.get)
                target_weights.loc[date, best_def] = 1.0
                selected_assets = [best_def]
                for t in def_scores:
                    log_details[t]['status'] = '✅ 편입' if t == best_def else '순위 미달'
            else:
                if defensive_assets:
                    all_def_scores = {t: mom_all.get(t, -np.inf) if hasattr(mom_all, 'get') else (mom_all[t] if t in mom_all.index else -np.inf) for t in defensive_assets}
                    best_fallback = max(all_def_scores, key=all_def_scores.get)
                    target_weights.loc[date, best_fallback] = 1.0
                    selected_assets = [best_fallback]
                    log_details[best_fallback] = {'score': all_def_scores[best_fallback], 'category': '방어', 'status': '✅ 편입 (예비)'}

            phase_scores_log[date] = {'phase': phase, 'details': log_details, 'selected': selected_assets}
            continue

        # ── 강세/중립 국면 ───────────────────────────────────────────────
        investment_mode[date] = '강세 (Aggressive)' if phase == '강세' else '중립 (Neutral)'
        mom_scores = calculate_acore_momentum(date, prices, config, phase)

        sharpe_scores = {}
        for ticker in aggressive_assets:
            mom_score = mom_scores.get(ticker, np.nan) if hasattr(mom_scores, 'get') else (mom_scores[ticker] if ticker in mom_scores.index else np.nan)
            cat = get_category(ticker)
            vol = get_vol(ticker, date)
            raw_sharpe = (mom_score / vol) if not pd.isna(vol) and vol > 0 else np.nan

            if pd.isna(mom_score) or mom_score <= 0:
                log_details[ticker] = {'score': raw_sharpe, 'category': cat, 'status': '모멘텀 음수'}
                continue

            if not calculate_volatility_zscore_filter(date, prices, ticker, vol_window, zscore_threshold):
                log_details[ticker] = {'score': raw_sharpe, 'category': cat, 'status': '과변동성 탈락'}
                continue

            if pd.isna(vol) or vol <= 0:
                log_details[ticker] = {'score': np.nan, 'category': cat, 'status': '데이터 부족'}
                continue

            if cat == '안전자산':
                continue

            sharpe_scores[ticker] = raw_sharpe
            log_details[ticker] = {'score': raw_sharpe, 'category': cat, 'status': '후보'}

        if not sharpe_scores:
            def_scores = {}
            for ticker in defensive_assets:
                ms = mom_scores.get(ticker, np.nan) if hasattr(mom_scores, 'get') else (mom_scores[ticker] if ticker in mom_scores.index else np.nan)
                if not pd.isna(ms) and ms > 0:
                    def_scores[ticker] = ms
            
            selected_assets = []
            if def_scores:
                best_def = max(def_scores, key=def_scores.get)
                target_weights.loc[date, best_def] = 1.0
                selected_assets = [best_def]
            elif defensive_assets:
                all_def_ms = {t: (mom_scores.get(t, -np.inf) if hasattr(mom_scores, 'get') else (mom_scores[t] if t in mom_scores.index else -np.inf)) for t in defensive_assets}
                best_fb = max(all_def_ms, key=all_def_ms.get)
                target_weights.loc[date, best_fb] = 1.0
                selected_assets = [best_fb]
            
            phase_scores_log[date] = {'phase': phase, 'details': log_details, 'selected': selected_assets}
            continue

        # 카테고리 캡 적용
        sorted_by_sharpe = sorted(sharpe_scores.items(), key=lambda x: x[1], reverse=True)
        category_count = {}
        filtered_scores = {}
        for ticker, score in sorted_by_sharpe:
            cat = get_category(ticker)
            cnt = category_count.get(cat, 0)
            if cnt < category_cap:
                filtered_scores[ticker] = score
                category_count[cat] = cnt + 1
            else:
                log_details[ticker]['status'] = '카테고리 캡 초과'

        if not filtered_scores:
            phase_scores_log[date] = {'phase': phase, 'details': log_details, 'selected': []}
            continue

        # Top N 선정
        top_sorted = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        selected_assets = [t for t, _ in top_sorted]

        weight_per = 1.0 / len(selected_assets)
        for asset in selected_assets:
            target_weights.loc[date, asset] = weight_per

        for t in filtered_scores:
            log_details[t]['status'] = '✅ 편입' if t in selected_assets else '순위 미달'

        phase_scores_log[date] = {
            'phase': phase,
            'details': log_details,
            'selected': selected_assets
        }

    return target_weights, investment_mode, phase_scores_log


def calculate_signals(prices, config):
    prices_copy = prices.copy()
    day_option = 'last' if config['rebalance_day'] == '월말' else 'first'
    if config['rebalance_freq'] == '분기별':
        prices_copy['year_quarter'] = prices_copy.index.to_period('Q').strftime('%Y-Q%q')
        rebal_dates = prices_copy.drop_duplicates('year_quarter', keep=day_option).index
    else: # 월별
        prices_copy['year_month'] = prices_copy.index.strftime('%Y-%m')
        rebal_dates = prices_copy.drop_duplicates('year_month', keep=day_option).index

    mom_type = config['momentum_params']['type']

    if mom_type == '13612U':
        mom_periods = [1, 3, 6, 12]
    else:
        mom_periods = config['momentum_params']['periods']

    if mom_type in ['13612U', '평균 모멘텀']:
        # 벡터화: 각 기간별 수익률을 한 번에 계산 후 평균
        returns_list = []
        prices_rebal = prices.reindex(rebal_dates)
        for month in mom_periods:
            # 거래일 기준 근사: 1개월 ≈ 21일 → prices에서 직접 shift
            past_prices = prices.shift(month * 21).reindex(rebal_dates)
            period_return = (prices_rebal / past_prices.replace(0, np.nan)) - 1
            period_return = period_return.fillna(0)
            returns_list.append(period_return)
        if returns_list:
            momentum_scores = sum(returns_list) / len(returns_list)
        else:
            momentum_scores = pd.DataFrame(0.0, index=rebal_dates, columns=prices.columns)

    elif mom_type == '상대 모멘텀':
        if not mom_periods:
            st.error("모멘텀 기간이 설정되지 않았습니다.")
            return pd.DataFrame()
        period_days = mom_periods[0] * 21
        momentum_scores = prices.pct_change(periods=period_days)
        momentum_scores = momentum_scores.reindex(rebal_dates).fillna(0)

    else:
        momentum_scores = pd.DataFrame(0.0, index=rebal_dates, columns=prices.columns)
            
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
    series = series.dropna()
    if series.empty or len(series) < 2:
        today = pd.Timestamp.today().normalize()
        return 0.0, today, today
    rolling_max = series.cummax()
    drawdown = (series - rolling_max) / rolling_max.replace(0, np.nan)
    drawdown = drawdown.fillna(0)
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


st.title("📈 [Quantest] 퀀트 백테스트_v1.0.0")

run_button_clicked = st.button("백테스트 실행", type="primary")
if st.session_state.get('settings_changed', False) and not run_button_clicked:
    st.warning("⚙️ 설정이 변경되었습니다. 백테스트를 다시 실행하여 최신 결과를 확인하세요!")

# '백테스트 실행' 버튼을 다시 생성하고, 모든 계산/실행 로직을 이 버튼 안으로 이동
if run_button_clicked:
    st.session_state.source = 'new_run'
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    st.session_state.uploader_key += 1
    
    # 2. 상태 업데이트 로직을 블록의 맨 앞으로 이동
    #    이렇게 하면 이 블록이 실행되는 즉시 '변경됨' 상태가 해제됩니다.
    st.session_state.last_run_config = current_config # 사이드바에서 이미 만든 current_config 사용
    st.session_state.settings_changed = False
    st.session_state.toast_shown = False
    
    # --- 여기서부터는 기존의 백테스트 실행 코드와 거의 동일합니다 ---
    
    # config 변수를 current_config로 대체하거나 그대로 사용
    config = current_config 
    
    all_tickers = sorted(list(set(aggressive_tickers + defensive_tickers + canary_tickers + [benchmark_ticker])))
    
    # A-Core 전략 시 S&P500 지수도 데이터 다운로드 대상에 추가
    if config.get('strategy_mode') == 'A-Core' and config.get('acore_config'):
        sp500_tk = config['acore_config'].get('sp500_ticker', '^GSPC')
        if sp500_tk not in all_tickers:
            all_tickers.append(sp500_tk)
    
    if any(ticker.endswith('.KS') for ticker in all_tickers):
        currency_symbol = '₩'
    else:
        currency_symbol = '$'
    
    
    with st.spinner('데이터 로딩 및 백테스트 실행 중...'):
        # 1. 실제 데이터 요청 시작일을 동적으로 계산
        # 모멘텀 계산에 필요한 최대 기간을 확인합니다.
        mom_type = config['momentum_params']['type']
        mom_periods = config['momentum_params']['periods']

        if mom_type == '13612U':
            # 13612U는 최대 12개월 수익률을 사용합니다.
            max_momentum_period = 12
        elif mom_periods:
            # '평균 모멘텀' 또는 '상대 모멘텀'의 경우, 설정된 기간 중 가장 긴 값을 사용합니다.
            max_momentum_period = max(mom_periods)
        else:
            # 예외적인 경우 (기간이 설정되지 않음)를 대비해 기본값 12개월을 사용합니다.
            max_momentum_period = 12

        # 백테스트 시작일로부터 최대 모멘텀 기간만큼 이전 날짜를 데이터 요청 시작일로 설정합니다.
        data_fetch_start_date = pd.to_datetime(config['start_date']) - pd.DateOffset(months=max_momentum_period)
        
        # 2. 계산된 시작일로 데이터를 요청합니다.
        prices, failed_tickers, culprit_tickers = get_price_data(all_tickers, data_fetch_start_date, config['end_date'], config['start_date'])
                
        if prices is None:
            st.error("데이터 로딩에 실패하여 백테스트를 중단합니다.")
            st.stop()

        momentum_scores = calculate_signals(prices, config)
        if momentum_scores.empty: st.error("모멘텀 시그널 계산에 실패했습니다."); st.stop()
        
        # ── 전략 분기: HAA vs A-Core ──────────────────────────────────────
        phase_scores_log = None
        market_phase_series = None

        if config.get('strategy_mode') == 'A-Core' and config.get('acore_config'):
            with st.spinner('🌡️ A-Core: 시장 국면 판단 중...'):
                market_phase_series = determine_market_phase(prices, canary_tickers, config)
            with st.spinner('📊 A-Core: 샤프비율 랭킹 & 카테고리 캡 적용 중...'):
                target_weights, investment_mode, phase_scores_log = construct_acore_portfolio(
                    prices, config, prices.columns.tolist(), market_phase_series
                )
                
                # --- [수정] A-Core: 표와 그래프에 샤프비율(모멘텀/변동성) 점수가 나오도록 덮어쓰기 ---
                dynamic_sharpe_scores = pd.DataFrame(index=market_phase_series.index, columns=prices.columns, dtype=float)
                vol_window = int(config['acore_config'].get('vol_window', 60)) # 설정된 변동성 기간 사용
                
                for date in market_phase_series.index:
                    phase = market_phase_series.get(date, '약세')
                    # 1. 국면별 모멘텀(수익률) 계산
                    moms = calculate_acore_momentum(date, prices, config, phase)
                    
                    for ticker in prices.columns:
                        # 2. 각 종목별 변동성 계산 (설정된 영업일 기준)
                        price_hist = prices[ticker].loc[:date]
                        if len(price_hist) > vol_window:
                            daily_ret = price_hist.pct_change().dropna().iloc[-vol_window:]
                            vol = daily_ret.std() * np.sqrt(252)
                            
                            if vol > 0:
                                # 3. 샤프비율 = 모멘텀 / 변동성
                                mom_val = moms.get(ticker, 0)
                                dynamic_sharpe_scores.loc[date, ticker] = mom_val / vol
                            else:
                                dynamic_sharpe_scores.loc[date, ticker] = 0
                        else:
                            dynamic_sharpe_scores.loc[date, ticker] = 0
                            
                # 최종적으로 샤프비율 점수를 화면 출력용 변수에 할당
                momentum_scores = dynamic_sharpe_scores 
        else:
            target_weights, investment_mode = construct_portfolio(momentum_scores, config, prices.columns.tolist())
        # ─────────────────────────────────────────────────────────────────
        
        returns_freq = config['backtest_type'].split(' ')[0]
        if returns_freq == '월별':
            rebal_dates = momentum_scores.index
            prices_rebal = prices.loc[rebal_dates]
            returns_rebal = prices_rebal.pct_change()
            turnover = (target_weights.shift(1) - target_weights).abs().sum(axis=1) / 2
            costs = turnover * config['transaction_cost']
            portfolio_returns = (target_weights.shift(1) * returns_rebal).sum(axis=1) - costs
            portfolio_returns = portfolio_returns.fillna(0)
            benchmark_returns = returns_rebal[config['benchmark']].fillna(0) if config['benchmark'] in returns_rebal.columns else pd.Series(0.0, index=returns_rebal.index)
        else: # 일별
            daily_weights = target_weights.reindex(prices.index, method='ffill').fillna(0)
            rebal_dates_series = pd.Series(index=prices.index, data=False)
            rebal_dates_series.loc[target_weights.index] = True
            turnover = (daily_weights.shift(1) - daily_weights).abs().sum(axis=1) / 2
            costs = turnover * config['transaction_cost']
            daily_returns = prices.pct_change().fillna(0)
            portfolio_returns = (daily_weights.shift(1) * daily_returns).sum(axis=1) - costs.where(rebal_dates_series, 0)
            benchmark_returns = daily_returns[config['benchmark']] if config['benchmark'] in daily_returns.columns else pd.Series(0.0, index=daily_returns.index)

        # 워밍업 기간(사전 로딩 기간)의 수익률 데이터를 제거합니다.
        start_date_dt = pd.to_datetime(config['start_date'])
        portfolio_returns = portfolio_returns[portfolio_returns.index >= start_date_dt]
        benchmark_returns = benchmark_returns[benchmark_returns.index >= start_date_dt]
        
        contribution_dates = target_weights.index[target_weights.index >= start_date_dt]
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
        mdd_start = mdd_end = bm_mdd_start = bm_mdd_end = pd.Timestamp('1970-01-01')
        if years > 0:
            cagr = (cumulative_returns.iloc[-1]/initial_cap)**(1/years) - 1
            bm_cagr = (benchmark_cumulative.iloc[-1]/initial_cap)**(1/years) - 1
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

        # 실제 백테스트 기간 내 리밸런싱 날짜 기준으로 추가 입금 횟수 계산
        start_date_dt = pd.to_datetime(config['start_date'])
        rebal_in_period = target_weights.index[target_weights.index >= start_date_dt]
        # 첫 번째 리밸런싱은 초기 투자이므로, 추가 입금은 두 번째부터
        num_contributions = max(0, len(rebal_in_period) - 1)
        total_contribution = config['initial_capital'] + (config['monthly_contribution'] * num_contributions)
        
        st.session_state['results'] = {
            'prices': prices, 'failed_tickers': failed_tickers, 'culprit_tickers': culprit_tickers,
            'max_momentum_period': max_momentum_period,
            'config': config, 'currency_symbol': currency_symbol, 'etf_df': etf_df,
            'momentum_scores': momentum_scores,
            'market_phase_series': market_phase_series,
            'phase_scores_log': phase_scores_log,
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
                'total_contribution': total_contribution,
                'total_profit': cumulative_returns.iloc[-1] - total_contribution,
                'cagr': cagr, 'mdd': mdd, 'mdd_start': mdd_start, 'mdd_end': mdd_end,
                'volatility': volatility, 'sharpe_ratio': sharpe_ratio, 'win_rate': win_rate,
                'bm_final_assets': benchmark_cumulative.iloc[-1],
                'bm_total_contribution': total_contribution,
                'bm_total_profit': benchmark_cumulative.iloc[-1] - total_contribution,
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

    if 'last_uploaded_file_id' in st.session_state:
        del st.session_state['last_uploaded_file_id']

    st.rerun()

# --- 탭과 결과 표시는 '백테스트 실행' 버튼 블록 바깥에 위치 ---
tab1, tab2 = st.tabs(["🚀 새로운 백테스트 결과", "📊 저장된 결과 비교"])

with tab1:
    st.header("🚀 백테스트 결과")
    st.divider()

    # --- .pkl 파일 업로드 기능 ---
    st.subheader("저장된 .pkl 파일 결과 보기")
    # --- [수정] 파일 업로더의 key를 동적으로 변경하도록 설정 ---
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_file_tab1 = st.file_uploader(
        "상세 결과를 보고 싶은 .pkl 파일을 업로드하세요.",
        type=['pkl'],
        key=f"uploader_tab1_{st.session_state.uploader_key}" # key를 동적으로 만듭니다.
    )

    if uploaded_file_tab1 is not None:
        # --- [추가] 새로운 백테스트 실행 직후에는 파일 로딩을 방지하는 조건 ---
        if st.session_state.get('source') != 'new_run':
            current_file_id = f"{uploaded_file_tab1.name}-{uploaded_file_tab1.size}"
            
            if current_file_id != st.session_state.get('last_uploaded_file_id'):
                try:
                    loaded_data = pickle.load(uploaded_file_tab1)
                    st.session_state['results'] = loaded_data
                    st.session_state.last_uploaded_file_id = current_file_id
                    st.session_state.source = 'file'
                    
                    # pkl 파일 안의 설정(config)을 불러와서 사이드바에 적용하도록 합니다.
                    if 'config' in loaded_data:
                        st.session_state.config_to_load = loaded_data['config']

                    st.session_state.toast_message = f"**'{uploaded_file_tab1.name}' 불러오기 완료**\n사이드바의 티커와 자산 목록이 업데이트 되었습니다."
                    st.rerun()  
                except Exception as e:
                    st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

    # --- [수정] 새로운 백테스트 실행 후에는 source 상태를 초기화 ---
    # 이 코드는 if uploaded_file_tab1 블록 바깥에 위치해야 합니다.
    if st.session_state.get('source') == 'new_run':
        st.session_state.source = None
    
    st.divider()

    # --- 결과 표시 로직 (기존 로직을 session_state 확인 후 실행하도록 변경) ---
    # session_state에 결과가 있을 경우 (새로 실행했거나, 불러왔거나)
    if 'results' in st.session_state and st.session_state['results']:
        results = st.session_state['results']
        
        # 불러온 결과의 이름 표시
        st.subheader(f"📑 결과 요약: {results.get('name', '신규 백테스트')}")
        
        prices = results['prices']
        failed_tickers = results['failed_tickers']
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
        

        st.header("1. 데이터 로딩 정보")
    
        # --- [수정] 모든 시나리오를 처리하기 위한 메시지 생성 로직 ---
    
        # 1. 세 가지 핵심 날짜를 정의합니다.
        user_start_date = pd.to_datetime(config['start_date'])
        user_start_date_str = user_start_date.strftime('%Y-%m-%d')
        
        data_load_start_date_str = prices.index[0].strftime('%Y-%m-%d')
        
        # 실제 분석이 시작되는 첫 거래일을 찾습니다.
        # prices 데이터에 사용자 시작일 이후의 날짜가 없을 경우를 대비해 예외 처리 추가
        analysis_start_index = prices.index[prices.index >= user_start_date]
        if not analysis_start_index.empty:
            analysis_start_date = analysis_start_index[0]
            analysis_start_date_str = analysis_start_date.strftime('%Y-%m-%d')
        else:
            # 이 경우는 데이터가 사용자 시작일 이전에 끝나버린 예외적인 상황
            analysis_start_date = prices.index[0]  # fallback: 데이터 첫날
            analysis_start_date_str = "데이터 없음"
    
        # 1단계: 백테스트 시작 사유에 대한 기본 메시지 표시
        # (이전과 동일한 if/elif/else 로직)
        if culprit_tickers:
            culprit_names = []
            for ticker in culprit_tickers:
                name = ticker
                if etf_df is not None:
                    match = etf_df[etf_df['Ticker'] == ticker]
                    if not match.empty:
                        name = match.iloc[0]['Name']
                culprit_names.append(f"'{name}'({ticker})")
        
            if len(culprit_tickers) == 1:
                culprits_str = culprit_names[0]
                reason_str = "의 데이터가 가장 늦게 시작되어"
            else:
                culprits_str = ', '.join(culprit_names)
                reason_str = " 등의 데이터가 가장 늦게 시작되어"
        
            st.warning(f"⚠️ {culprits_str} {reason_str}, 모든 자산이 존재하는 **{data_load_start_date_str}**부터 백테스트를 시작합니다.")
        elif data_load_start_date_str < user_start_date_str:
            st.info(
                f"💡 정확한 모멘텀 계산을 위해 **{data_load_start_date_str}**부터 데이터를 미리 불러왔습니다.\n\n"
                f"실제 백테스트와 모든 성과 분석은 요청하신 기간의 첫 거래일인 **{analysis_start_date_str}**부터 시작됩니다."
            )
        elif analysis_start_date_str > user_start_date_str:
            st.info(f"💡 요청하신 기간의 첫 거래일인 **{analysis_start_date_str}**부터 백테스트를 시작합니다.")
        else:
            st.success(f"✅ 백테스트가 설정하신 시작일인 **{user_start_date_str}**에 맞춰 정상적으로 시작됩니다.")
    
        # 2단계: 워밍업 기간이 충분했는지 독립적으로 확인하고, 필요 시 추가 안내
        max_momentum_period = results.get('max_momentum_period', 12)
        data_load_start_date = prices.index[0]
        
        # 실제 워밍업 기간을 월 단위로 계산
        available_warmup_months = (analysis_start_date.year - data_load_start_date.year) * 12 + (analysis_start_date.month - data_load_start_date.month)
    
        if available_warmup_months < max_momentum_period:
            st.info(f"💡 **참고:** 설정된 최대 모멘텀 기간({max_momentum_period}개월)보다 실제 데이터 기간이 짧아, 백테스트 초기에는 불완전한 모멘텀 점수가 사용됩니다.")

        if failed_tickers: 
            st.warning(f"다운로드에 실패한 티커가 있습니다: {', '.join(failed_tickers)}")    
        
        with st.expander("데이터 미리보기 (최근 5일)"):
            display_df = prices.tail().copy()
            new_column_names = []
            for ticker in display_df.columns:
                full_name = ticker
                if etf_df is not None:
                    match = etf_df[etf_df['Ticker'] == ticker]
                    if not match.empty: 
                        full_name = match.iloc[0]['Name']
                new_column_names.append(full_name)
            display_df.columns = new_column_names
            st.dataframe(display_df.style.format("{:,.2f}"))
            
        st.subheader("사용한 자산군 정보")
        config_tickers = config.get('tickers', {})
        
        # --- [추가] 벤치마크 정보 표시 ---
        benchmark_ticker = config.get('benchmark')
        if benchmark_ticker:
            st.markdown("**벤치마크**")
            benchmark_name = benchmark_ticker
            if etf_df is not None:
                match = etf_df[etf_df['Ticker'] == benchmark_ticker]
                if not match.empty:
                    benchmark_name = match.iloc[0]['Name']
            
            display_benchmark = f"{benchmark_ticker} - {benchmark_name}" if benchmark_ticker != benchmark_name else benchmark_ticker
            
            # --- [수정] 아래 div의 style에 margin-bottom을 추가하여 간격을 줍니다 ---
            st.markdown(
                f'<div style="background-color: #f0f2f6; border-radius: 0.25rem; padding: 0.75rem; color: #31333F; margin-bottom: 1rem;">{display_benchmark}</div>', 
                unsafe_allow_html=True
            )
        
        # 티커 리스트를 '티커 - 전체이름' 형식의 문자열 리스트로 변환하는 헬퍼 함수
        def format_asset_list(ticker_list, df):
            if not ticker_list:
                return "없음"
            
            formatted_items = []
            for ticker in ticker_list:
                full_name = ticker  # 기본값은 티커로 설정
                if df is not None:
                    match = df[df['Ticker'] == ticker]
                    if not match.empty:
                        full_name = match.iloc[0]['Name']
                
                # 티커와 이름이 다를 경우에만 " - "로 연결
                display_item = f"{ticker} - {full_name}" if ticker != full_name else ticker
                formatted_items.append(display_item)
            
            # 각 항목을 쉼표와 줄바꿈으로 연결하여 가독성 향상
            return ", \n".join(formatted_items)

        # 각 자산군 목록을 가져와 포맷팅
        canary_list = config_tickers.get('CANARY', [])
        aggressive_list = config_tickers.get('AGGRESSIVE', [])
        defensive_list = config_tickers.get('DEFENSIVE', [])
        
        st.markdown("**카나리아**")
        st.info(format_asset_list(canary_list, etf_df))
        st.markdown("**공격 자산**")
        st.success(format_asset_list(aggressive_list, etf_df))
        st.markdown("**방어 자산**")
        st.warning(format_asset_list(defensive_list, etf_df))

        # [추가] 사용한 시그널 설정 정보 표시
        with st.expander("사용한 시그널 설정"):
            if config.get('strategy_mode') == 'A-Core':
                # ── A-Core 전략: 국면별 실제 시그널 공식 표시 ──────────────────
                st.markdown("**모멘텀 계산 기준 기간**: `1, 3, 6, 12개월` (고정)")
                st.markdown("**카나리아 판단**: `1·3·6·12개월 수익률 평균`으로 리스크 국면 결정")
                st.divider()

                st.markdown("##### 🟢 강세 국면 — 가중 모멘텀")
                st.markdown("""
| 기간 | 가중치 | 계산식 |
|:---:|:---:|:---|
| 1개월 | 40% | r₁ × 0.4 |
| 3개월 | 30% | r₃ × 0.3 |
| 6개월 | 20% | r₆ × 0.2 |
| 12개월 | 10% | r₁₂ × 0.1 |
| **합계** | **100%** | **score = r₁×0.4 + r₃×0.3 + r₆×0.2 + r₁₂×0.1** |
""")
                st.caption("→ 단기 모멘텀에 더 높은 비중을 부여해 상승 추세에 빠르게 반응합니다.")

                st.divider()
                st.markdown("##### 🟡 중립 국면 / 🔴 약세 국면 — 평균 모멘텀")
                st.markdown("""
| 기간 | 가중치 | 계산식 |
|:---:|:---:|:---|
| 1개월 | 25% | r₁ × 0.25 |
| 3개월 | 25% | r₃ × 0.25 |
| 6개월 | 25% | r₆ × 0.25 |
| 12개월 | 25% | r₁₂ × 0.25 |
| **합계** | **100%** | **score = (r₁ + r₃ + r₆ + r₁₂) / 4** |
""")
                st.caption("→ 전 기간 동일 비중으로 노이즈를 줄이고 안정적인 신호를 생성합니다.")

                st.divider()
                st.markdown("##### 📐 샤프비율 랭킹 (강세·중립 국면)")
                ac = config.get('acore_config', {})
                vol_window = ac.get('vol_window', 60)
                zscore_thr = ac.get('zscore_threshold', 1.5)
                st.markdown(f"""
- **샤프비율 점수** = 모멘텀 점수 ÷ 연율화 변동성
- **변동성 계산 기간**: `{vol_window}일` (연율화: ×√252)
- **절대 모멘텀 필터**: 점수 ≤ 0인 자산 제외
- **과변동성 Z-Score 필터**: 최근 1개월 변동성 > 12개월 평균 + (표준편차 × `{zscore_thr}`) 이면 제외
""")

            else:
                # ── HAA 전략: 기존 사이드바 설정 그대로 표시 ──────────────────
                momentum_params = config.get('momentum_params', {})
                mom_type = momentum_params.get('type', 'N/A')
                periods = momentum_params.get('periods', [])

                st.markdown(f"**모멘텀 종류**: `{mom_type}`")
                st.markdown(f"**모멘텀 기간**: `{', '.join(map(str, periods))}` (개월)")

                if mom_type == '13612U':
                    st.caption("→ 1·3·6·12개월 수익률의 단순 평균으로 모멘텀 점수를 계산합니다. (기간 입력값 무시)")
                elif mom_type == '평균 모멘텀':
                    st.caption(f"→ 입력된 {', '.join(map(str, periods))}개월 수익률을 동일 비중으로 평균합니다.")
                elif mom_type == '상대 모멘텀':
                    period_used = periods[0] if periods else 'N/A'
                    st.caption(f"→ {period_used}개월 수익률 기준으로 자산 간 상대 순위를 비교합니다.")

          # [추가] 사용한 포트폴리오 구성 전략 정보 표시
        with st.expander("사용한 포트폴리오 구성 전략"):
            portfolio_params = config.get('portfolio_params', {})
            use_canary = portfolio_params.get('use_canary', False)
            use_hybrid = portfolio_params.get('use_hybrid_protection', False)
            top_agg = portfolio_params.get('top_n_aggressive', 'N/A')
            top_def = portfolio_params.get('top_n_defensive', 'N/A')
            weighting = portfolio_params.get('weighting', 'N/A')
            st.markdown(f"**운용 전략**: `{config.get('strategy_mode', 'HAA')}`")
            st.markdown(f"**카나리아 자산 사용 (Risk-On/Off)**: `{'사용' if use_canary else '미사용'}`")
            st.markdown(f"**하이브리드 보호 장치 사용**: `{'사용' if use_hybrid else '미사용'}`")
            st.markdown(f"**공격 자산 Top N**: `{top_agg}`")
            st.markdown(f"**방어 자산 Top N**: `{top_def}`")
            st.markdown(f"**자산 배분 방식**: `{weighting}`")
            
            # A-Core 전용 설정 표시
            if config.get('strategy_mode') == 'A-Core' and config.get('acore_config'):
                ac = config['acore_config']
                st.divider()
                st.markdown("**A-Core 전략 설정**")
                st.markdown(f"- S&P500 티커: `{ac.get('sp500_ticker', '^GSPC')}`")
                st.markdown(f"- 이동평균 기간: `{ac.get('ma_period', 200)}일`")
                st.markdown(f"- 변동성 계산 기간: `{ac.get('vol_window', 60)}일`")
                st.markdown(f"- 카테고리 캡: `{ac.get('category_cap', 2)}`")
                st.markdown(f"- Z-Score 임계값: `{ac.get('zscore_threshold', 1.5)}`")     
            
        # 모든 메시지 표시 후, 분석 데이터를 시작일 기준으로 필터링
        backtest_start_date = pd.to_datetime(results['config']['start_date'])

        # results 딕셔너리 내부의 데이터를 직접 필터링
        results['prices'] = results['prices'][results['prices'].index >= backtest_start_date]
        results['momentum_scores'] = results['momentum_scores'][results['momentum_scores'].index >= backtest_start_date]
        results['target_weights'] = results['target_weights'][results['target_weights'].index >= backtest_start_date]
        results['investment_mode'] = results['investment_mode'][results['investment_mode'].index >= backtest_start_date]

        # 이후 코드에서 사용할 'prices' 변수도 필터링된 데이터로 다시 할당
        prices = results['prices']
        
        st.header("2. 시그널 모멘텀")
        
        # ── [A-Core 신규] 시장 국면 통계 카드 (차트와 분리하여 먼저 표시) ───
        if config.get('strategy_mode') == 'A-Core':
            market_phase_series = results.get('market_phase_series')
            phase_scores_log    = results.get('phase_scores_log')

            if market_phase_series is not None and not market_phase_series.empty:
                st.subheader("🌡️ A-Core 시장 국면 판단 결과")
                phase_counts = market_phase_series.value_counts()
                total = len(market_phase_series)
                col_bull, col_neut, col_bear = st.columns(3)
                with col_bull:
                    cnt = phase_counts.get('강세', 0)
                    st.metric("🟢 강세 국면", f"{cnt}회", f"{cnt/total:.1%}")
                with col_neut:
                    cnt = phase_counts.get('중립', 0)
                    st.metric("🟡 중립 국면", f"{cnt}회", f"{cnt/total:.1%}")
                with col_bear:
                    cnt = phase_counts.get('약세', 0)
                    st.metric("🔴 약세 국면", f"{cnt}회", f"{cnt/total:.1%}")

                # 샤프비율 점수 로그 상세 보기
                if phase_scores_log:
                    with st.expander("📊 A-Core 샤프비율 랭킹 상세 (최근 12회)"):
                        st.markdown("> **💡 참고:** 점수가 높아도 편입되지 않은 경우, 우측 `상태` 열에서 이유(카테고리 캡 초과, 과변동성 탈락 등)를 명확히 확인할 수 있습니다.")
                        recent_dates = sorted(phase_scores_log.keys(), reverse=True)[:12]
                        for d in recent_dates:
                            log = phase_scores_log[d]
                            phase_str = log.get('phase', '-')
                            details   = log.get('details', {})
                            selected  = log.get('selected', [])
                            
                            if not details:
                                st.markdown(f"**{d.strftime('%Y-%m')}** [{phase_str}] → 편입 자산 없음 (현금)")
                                continue
                                
                            rows = []
                            for tk, info in sorted(details.items(), key=lambda x: (pd.isna(x[1]['score']), -x[1]['score'] if not pd.isna(x[1]['score']) else 0)):
                                name = tk
                                if etf_df is not None:
                                    m = etf_df[etf_df['Ticker'] == tk]
                                    if not m.empty:
                                        name = f"{tk} - {m.iloc[0]['Name']}"
                                
                                score_val = info.get('score', np.nan)
                                score_str = f"{score_val:.3f}" if not pd.isna(score_val) else "-"
                                
                                rows.append({
                                    '자산': name, 
                                    '카테고리': info.get('category', '-'),
                                    '점수': score_str,
                                    '상태': info.get('status', '-')
                                })
                                
                            df_log = pd.DataFrame(rows)
                            st.markdown(f"**{d.strftime('%Y-%m')}** [{phase_str}] → 선택: {', '.join(selected) if selected else '방어 자산 대피/현금'}")
                            
                            # 👇 height=210을 추가하여 약 5줄 정도만 보이도록 스크롤을 생성합니다.
                            st.dataframe(df_log.set_index('자산'), use_container_width=True, height=210)

            st.divider()

        # --- 카나리아 모멘텀 vs 벤치마크 가격 비교 그래프 ---
        st.subheader("📊 카나리아 모멘텀 추이 vs. 벤치마크 가격")

        # 파스텔 톤 팔레트 정의
        _PASTEL = {
            '강세': '#c8f0c8',   # 연초록
            '중립': '#fff9c4',   # 연노랑
            '약세': '#ffd6d6',   # 연분홍
            'risk_on':  '#d4edda',  # HAA Aggressive
            'risk_off': '#fff3cd',  # HAA Defensive
        }

        prices_for_chart = results.get('prices')
        config_for_chart = results.get('config')

        if prices_for_chart is None or config_for_chart is None:
            st.warning("그래프를 그리는데 필요한 데이터(가격, 설정)가 결과에 포함되지 않았습니다.")
        else:
            full_momentum_scores = calculate_full_momentum(prices_for_chart, config_for_chart)

            _bt_type    = config_for_chart.get('backtest_type', '일별')
            _rebal_day  = config_for_chart.get('rebalance_day', '월말')

            if _bt_type == '월별':
                if _rebal_day == '월초':
                    display_momentum = full_momentum_scores.resample('MS').first()
                    display_prices   = prices_for_chart.resample('MS').first()
                else:
                    # pandas 2.2+: 'M' → 'ME'
                    display_momentum = full_momentum_scores.resample('ME').last()
                    display_prices   = prices_for_chart.resample('ME').last()
            else:
                display_momentum = full_momentum_scores
                display_prices   = prices_for_chart

            _canary_tickers  = config_for_chart['tickers']['CANARY']
            _benchmark_ticker = config_for_chart['benchmark']

            if _canary_tickers and _benchmark_ticker in display_prices.columns:
                canary_momentum = display_momentum[
                    [t for t in _canary_tickers if t in display_momentum.columns]
                ].mean(axis=1)
                benchmark_price = display_prices[_benchmark_ticker]

                fig_mom, ax_mom = plt.subplots(figsize=(10, 5))
                ax_price = ax_mom.twinx()

                # ── 배경 국면/모멘텀 음영 ────────────────────────────────────
                is_acore = config_for_chart.get('strategy_mode') == 'A-Core'
                _market_phase_s = results.get('market_phase_series')

                if is_acore and _market_phase_s is not None and not _market_phase_s.empty:
                    # A-Core: 3단계 국면 배경 (파스텔)
                    _phase_changes = _market_phase_s.loc[
                        _market_phase_s.shift(1) != _market_phase_s
                    ].index.tolist()
                    if _market_phase_s.index[0] not in _phase_changes:
                        _phase_changes.insert(0, _market_phase_s.index[0])
                    for _pi in range(len(_phase_changes)):
                        _ps = _phase_changes[_pi]
                        _pe = _phase_changes[_pi+1] if _pi+1 < len(_phase_changes) else canary_momentum.index[-1]
                        _ph = _market_phase_s.loc[_ps]
                        _pc = _PASTEL.get(_ph, '#eeeeee')
                        ax_mom.axvspan(_ps, _pe, facecolor=_pc, alpha=0.6, zorder=0)
                else:
                    # HAA: 카나리아 양/음 구간 배경을 누적 수익 그래프와 완벽히 통일
                    is_positive = canary_momentum >= 0
                    mode_changes = is_positive.loc[is_positive.shift(1) != is_positive].index.tolist()
                    if is_positive.index[0] not in mode_changes: 
                        mode_changes.insert(0, is_positive.index[0])
                        
                    for _i in range(len(mode_changes)):
                        _start = mode_changes[_i]
                        _end = mode_changes[_i+1] if _i+1 < len(mode_changes) else canary_momentum.index[-1]
                        _is_on = is_positive.loc[_start]
                        _color = _PASTEL['risk_on'] if _is_on else _PASTEL['risk_off']
                        ax_mom.axvspan(_start, _end, facecolor=_color, alpha=0.6, zorder=0)

                # ── 카나리아 모멘텀 라인 ─────────────────────────────────────
                ax_mom.plot(canary_momentum.index, canary_momentum,
                            label=f'Canary Momentum ({", ".join(_canary_tickers)})',
                            color='steelblue', linewidth=1.5, zorder=3)
                ax_mom.axhline(0, color='crimson', linestyle=':', linewidth=1.2, zorder=2)
                ax_mom.set_ylabel('카나리아 모멘텀 점수', fontsize=12)
                ax_mom.yaxis.set_major_formatter(
                    mtick.FuncFormatter(lambda y, _: f'{y:.1%}')
                )

                # ── 벤치마크 가격 라인 ───────────────────────────────────────
                ax_price.plot(benchmark_price.index, benchmark_price,
                              label=f'Benchmark Price ({_benchmark_ticker})',
                              color='dimgray', linewidth=1.0, alpha=0.8, zorder=2)
                ax_price.set_ylabel(f'{_benchmark_ticker} 가격', fontsize=12)

                # ── 범례 통합 ────────────────────────────────────────────────
                lines1, labels1 = ax_mom.get_legend_handles_labels()
                lines2, labels2 = ax_price.get_legend_handles_labels()

                if is_acore:
                    legend_patches = [
                        Patch(facecolor=_PASTEL['강세'],  label='강세 국면'),
                        Patch(facecolor=_PASTEL['중립'],  label='중립 국면'),
                        Patch(facecolor=_PASTEL['약세'],  label='약세 국면'),
                    ]
                else:
                    # 누적 수익 그래프와 범례 텍스트 완벽 일치
                    legend_patches = [
                        Patch(facecolor=_PASTEL['risk_on'], label='Risk-On (카나리아 양수)'),
                        Patch(facecolor=_PASTEL['risk_off'], label='Risk-Off (방어)'),
                    ]

                ax_mom.legend(lines1 + lines2 + legend_patches,
                              labels1 + labels2 + [p.get_label() for p in legend_patches],
                              loc='upper left', fontsize=9)

                ax_mom.set_title('카나리아 모멘텀 vs. 벤치마크 가격', fontsize=16)
                ax_mom.set_xlabel('날짜', fontsize=12)
                ax_mom.grid(True, which='both', ls='--', linewidth=0.4, alpha=0.6)
                fig_mom.tight_layout()
                st.pyplot(fig_mom)
                plt.close(fig_mom)
            else:
                st.warning("카나리아 또는 벤치마크 자산 데이터를 찾을 수 없습니다.")

        # --- [수정] 구성종목 모멘텀 점수 (중복 컬럼 에러 및 KeyError 방지) ---
        # 전략 종류에 따라 라벨 텍스트(샤프비율 vs 모멘텀)를 동적으로 변경
        is_acore_strat = config.get('strategy_mode') == 'A-Core'
        score_label = '샤프비율 점수' if is_acore_strat else '모멘텀 점수'
        
        st.subheader(f"📊 구성종목 {score_label}")

        momentum_scores = results.get('momentum_scores')
        config = results.get('config')

        if momentum_scores is not None and config is not None:
            # 1. 공격/방어 자산 목록을 가져옵니다.
            aggressive_tickers = config['tickers']['AGGRESSIVE']
            defensive_tickers = config['tickers']['DEFENSIVE']
            
            # 2. 두 리스트를 합친 후, 중복을 제거하여 고유한 티커 목록을 만듭니다.
            combined_assets = aggressive_tickers + defensive_tickers
            unique_assets = list(dict.fromkeys(combined_assets))
            
            # 3. 모멘텀 점수 데이터에 실제 존재하는 티커만 필터링합니다.
            assets_to_show = [t for t in unique_assets if t in momentum_scores.columns]
            
            if assets_to_show:
                scores_to_display = momentum_scores[assets_to_show]

                # 데이터 테이블 (기존과 동일)
                with st.expander(f"{score_label} 상세 데이터 보기 (전체 기간)"):
                    #end_date = scores_to_display.index.max()
                    #start_date = end_date - pd.DateOffset(months=12)
                    #recent_scores = scores_to_display[scores_to_display.index >= start_date]
                    #sorted_recent_scores = recent_scores.sort_index(ascending=False)
                    sorted_recent_scores = scores_to_display.sort_index(ascending=False)
                    
                    if not sorted_recent_scores.empty:
                        df_to_display = sorted_recent_scores.copy()
                        
                        # Stock_list.csv 정보가 있을 경우, 컬럼 이름을 전체 이름으로 변경
                        if etf_df is not None:
                            # Ticker를 키로, Name을 값으로 하는 딕셔너리 생성
                            ticker_to_name_map = pd.Series(etf_df.Name.values, index=etf_df.Ticker).to_dict()
                            df_to_display.rename(columns=ticker_to_name_map, inplace=True)

                        # [수정] 3자리 소수점 일반 숫자로 포맷 변경
                        st.dataframe(df_to_display.style.format("{:.3f}").background_gradient(cmap='viridis', axis=1))
                    else:
                        st.dataframe(sorted_recent_scores)

                # 1. 데이터를 'long' 형태로 변환
                df_melted = scores_to_display.reset_index().rename(columns={'index': 'Date'})
                df_melted = df_melted.melt(id_vars='Date', var_name='Ticker', value_name='Momentum Score')

                # 2. Stock_list.csv의 이름 정보를 df_melted에 합치기(merge)
                if etf_df is not None:
                    # Ticker를 기준으로 이름(Name) 컬럼을 추가합니다.
                    df_merged = pd.merge(
                        df_melted, 
                        etf_df[['Ticker', 'Name']], 
                        on='Ticker', 
                        how='left' # 모멘텀 데이터 기준으로 합치기
                    )
                else:
                    # Stock_list.csv가 없으면 Name 컬럼을 Ticker와 동일하게 설정
                    df_merged = df_melted.copy()
                    df_merged['Name'] = df_merged['Ticker']

                # 3. Plotly Express 라인 차트 생성 시 호버 옵션 추가
                fig_interactive = px.line(
                    df_merged, # 이름이 추가된 데이터프레임 사용
                    x='Date',
                    y='Momentum Score',
                    color='Name',
                    title=f'구성종목 {score_label} 추이',
                    labels={'Date': 'Date', 'Momentum Score': score_label, 'Name': '종목명'},
                    hover_name='Name', # 호버 툴팁의 제목을 'Name'으로 설정
                    custom_data=['Ticker']
                )
                
                # 4. 툴팁(hovertemplate) 서식 지정 (동적 라벨 적용)
                fig_interactive.update_traces(
                    hovertemplate=(
                        "<b>%{hovertext}</b><br><br>" + 
                        "티커: %{customdata[0]}<br>" +     
                        f"{score_label}: %{{y:.3f}}<br>" +      
                        "날짜: %{x|%Y-%m-%d}" +            
                        "<extra></extra>"                
                    )
                )

                # --- [추가] 방어 자산(안전자산)은 그래프에서 기본적으로 선이 숨겨지도록 설정 ---
                defensive_tickers_list = config['tickers']['DEFENSIVE']
                for trace in fig_interactive.data:
                    # trace.customdata의 첫 번째 항목(티커명)이 방어 자산 목록에 있는지 확인
                    if trace.customdata is not None and trace.customdata[0][0] in defensive_tickers_list:
                        trace.visible = 'legendonly' # 범례를 클릭해야만 선이 보이도록 설정
                # -----------------------------------------------------------------------------

                fig_interactive.add_hline(y=0, line_dash="dot", line_color="red")
                fig_interactive.update_layout(legend_title_text='종목명')
                
                st.plotly_chart(fig_interactive, use_container_width=True)
                
            else:
                st.info("표시할 공격 또는 방어 자산의 모멘텀 데이터가 없습니다.")
        else:
            st.warning("모멘텀 점수 데이터를 결과 파일에서 찾을 수 없습니다.")

        st.header("3. 백테스트 결과")        
        
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
              f"데이터: {config['backtest_type']} | "
              f"주기: {config['rebalance_freq']} | "
              f"기준일: {config['rebalance_day']} | "
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
                # metrics에 저장된 total_contribution에서 역산하여 횟수 표시
                _num_c = round((metrics['total_contribution'] - config['initial_capital']) / config['monthly_contribution']) if config['monthly_contribution'] > 0 else 0
                breakdown_str = f"(초기: {currency_symbol}{config['initial_capital']:,.0f} + 추가: {currency_symbol}{config['monthly_contribution']:,.0f} × {_num_c}회)"
                st.markdown(f"<p style='font-size: 0.8em; color: #555; margin-top: -10px;'>{breakdown_str}</p>", unsafe_allow_html=True)
            
            st.metric("CAGR (연평균 수익률)", f"{metrics['cagr']:.2%}", help="초기 투자금 기준 포트폴리오의 연평균 복리수익률입니다. (적립식 추가 입금 포함 전략 전체 성과 기준)")
            
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
                _num_c = round((metrics['bm_total_contribution'] - config['initial_capital']) / config['monthly_contribution']) if config['monthly_contribution'] > 0 else 0
                breakdown_str = f"(초기: {currency_symbol}{config['initial_capital']:,.0f} + 추가: {currency_symbol}{config['monthly_contribution']:,.0f} × {_num_c}회)"
                st.markdown(f"<p style='font-size: 0.8em; color: #555; margin-top: -10px;'>{breakdown_str}</p>", unsafe_allow_html=True)

            st.metric("CAGR (연평균 수익률)", f"{metrics['bm_cagr']:.2%}")

            bm_mdd_help = f"{metrics['bm_mdd_start'].strftime('%Y-%m-%d')} ~ {metrics['bm_mdd_end'].strftime('%Y-%m-%d')}"
            st.metric("MDD (최대 낙폭)", f"{metrics['bm_mdd']:.2%}", help=bm_mdd_help)

            st.metric("Volatility (변동성)", f"{metrics['bm_volatility']:.2%}")
            st.metric("Sharpe Ratio (샤프 지수)", f"{metrics['bm_sharpe_ratio']:.2f}")
            st.metric("Win Rate (승률)", f"{metrics['bm_win_rate']:.2%}")
        
        st.subheader("📊 누적 수익 그래프")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # --- [추가] 모멘텀 그래프와 완벽하게 동일한 파스텔 톤 팔레트 ---
        _PASTEL = {
            '강세': '#c8f0c8',   # 연초록
            '중립': '#fff9c4',   # 연노랑
            '약세': '#ffd6d6',   # 연분홍
            'risk_on':  '#d4edda',  # HAA Aggressive
            'risk_off': '#fff3cd',  # HAA Defensive
        }
        
        if not investment_mode.empty:
            mode_changes = investment_mode.loc[investment_mode.shift(1) != investment_mode].index.tolist()
            if investment_mode.index[0] not in mode_changes: mode_changes.insert(0, investment_mode.index[0])
            for i in range(len(mode_changes)):
                start_interval = mode_changes[i]
                end_interval = mode_changes[i+1] if i+1 < len(mode_changes) else cumulative_returns.index[-1]
                mode = investment_mode.loc[start_interval]
                
                # 색상 매칭
                if '강세' in mode: color = _PASTEL['강세']
                elif '중립' in mode: color = _PASTEL['중립']
                elif '약세' in mode: color = _PASTEL['약세']
                elif mode == 'Aggressive': color = _PASTEL['risk_on']
                else: color = _PASTEL['risk_off']
                
                # 투명도(alpha)도 0.6으로 통일
                ax.axvspan(start_interval, end_interval, facecolor=color, alpha=0.6, zorder=0)
                
        line1, = ax.plot(cumulative_returns.index, cumulative_returns, label='전략', color='royalblue', linewidth=1.0, zorder=3)
        line2, = ax.plot(benchmark_cumulative.index, benchmark_cumulative, label='벤치마크', color='grey', linewidth=1.0, zorder=2)
        
        first_valid_date = cumulative_returns.first_valid_index()
        last_valid_date = cumulative_returns.last_valid_index()

        if first_valid_date is not None and last_valid_date is not None:
            margin_days = (last_valid_date - first_valid_date).days * 0.05
            graph_start_date = first_valid_date - pd.DateOffset(days=margin_days)
            graph_end_date = last_valid_date + pd.DateOffset(days=margin_days)
            ax.set_xlim(left=graph_start_date, right=graph_end_date)      
        
        # --- [수정] 범례 텍스트도 모멘텀 그래프와 완전히 동일하게 변경 ---
        if config.get('strategy_mode') == 'A-Core':
            legend_handles = [
                line1, line2,
                Patch(facecolor=_PASTEL['강세'], label='강세 국면'),
                Patch(facecolor=_PASTEL['중립'], label='중립 국면'),
                Patch(facecolor=_PASTEL['약세'], label='약세 국면'),
            ]
        else:
            legend_handles = [
                line1, line2, 
                Patch(facecolor=_PASTEL['risk_on'], label='Risk-On (카나리아 양수)'), 
                Patch(facecolor=_PASTEL['risk_off'], label='Risk-Off (방어)') # HAA 방어 텍스트 매칭
            ]
            
        ax.set_title('누적 자산 추이', fontsize=16)
        ax.set_xlabel('날짜', fontsize=12); ax.set_ylabel('포트폴리오 가치', fontsize=12)
        formatter = mtick.FuncFormatter(lambda y, _: format_large_number(y, symbol=currency_symbol))
        ax.yaxis.set_major_formatter(formatter)
        ax.legend(handles=legend_handles, loc='upper left', fontsize=10); ax.grid(True, which="both", ls="--", linewidth=0.5)
        st.pyplot(fig)
        plt.close(fig)
        
        st.markdown("---")
        st.header("🔬 상세 분석")
        
        st.subheader("📅 연도별 수익률")
        col1_annual, col2_annual = st.columns([1, 2])
        returns_freq = config['backtest_type'].split(' ')[0]
        if returns_freq == '일별':
            monthly_pf_returns_for_annual = portfolio_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
            monthly_bm_returns_for_annual = benchmark_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        else:
            monthly_pf_returns_for_annual = portfolio_returns; monthly_bm_returns_for_annual = benchmark_returns
        annual_returns = monthly_pf_returns_for_annual.resample('YE').apply(lambda x: (1 + x).prod() - 1).to_frame(name="전략")
        bm_annual_returns = monthly_bm_returns_for_annual.resample('YE').apply(lambda x: (1 + x).prod() - 1).to_frame(name="벤치마크")
        annual_df = pd.concat([annual_returns, bm_annual_returns], axis=1)
        annual_df.index = annual_df.index.year
        annual_df.index = annual_df.index.astype(str)
        annual_df.index.name = "연도"  # 인덱스 이름 재설정
        with col1_annual: st.dataframe(annual_df.style.format("{:.2%}"))
        with col2_annual:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            annual_df.plot(kind='bar', ax=ax2, color=['royalblue', 'grey']); ax2.set_title('연도별 수익률', fontsize=16)
            ax2.set_xlabel('연도', fontsize=12); ax2.set_ylabel('수익률', fontsize=12); ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax2.tick_params(axis='x', rotation=45); ax2.grid(axis='y', linestyle='--', linewidth=0.5); st.pyplot(fig2); plt.close(fig2)

        st.subheader("📉 하락폭(Drawdown) 추이")
        # 데이터 타입 오류 방지를 위해 astype(float).fillna(0) 추가
        strategy_dd = (strategy_growth / strategy_growth.cummax() - 1).astype(float).fillna(0)
        benchmark_dd = (benchmark_growth / benchmark_growth.cummax() - 1).astype(float).fillna(0)
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(strategy_dd.index, strategy_dd, label='전략 하락폭', color='royalblue', linewidth=1.0)
        ax3.plot(benchmark_dd.index, benchmark_dd, label='벤치마크 하락폭', color='grey', linewidth=1.0)
        ax3.fill_between(strategy_dd.index, strategy_dd, 0, color='royalblue', alpha=0.1)
        ax3.set_title('하락폭(Drawdown) 추이', fontsize=16)
        ax3.set_xlabel('날짜', fontsize=12); ax3.set_ylabel('하락폭 (%)', fontsize=12); ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax3.legend(loc='lower right', fontsize=10); ax3.grid(True, which="both", ls="--", linewidth=0.5); st.pyplot(fig3); plt.close(fig3)
        
        st.subheader("🗓️ 월별 수익률 히트맵")
        if not monthly_pf_returns_for_annual.empty:
            heatmap_df = monthly_pf_returns_for_annual.to_frame(name='Return').copy()
            # 데이터 형식을 숫자로 강제 변환하여 HAA에서도 그라데이션 색상이 정상 적용되도록 수정
            heatmap_df['Return'] = pd.to_numeric(heatmap_df['Return'], errors='coerce').fillna(0)
            heatmap_df['Year'] = heatmap_df.index.year
            heatmap_df['Month'] = heatmap_df.index.month
            
            heatmap_pivot = heatmap_df.pivot_table(index='Year', columns='Month', values='Return', aggfunc='first')
            # 정수 월 번호를 한국어 월 이름으로 변환
            month_labels = {1:'1월',2:'2월',3:'3월',4:'4월',5:'5월',6:'6월',
                           7:'7월',8:'8월',9:'9월',10:'10월',11:'11월',12:'12월'}
            heatmap_pivot.columns = [month_labels.get(c, str(c)) for c in heatmap_pivot.columns]
            monthly_avg = heatmap_pivot.mean()
            heatmap_pivot.loc['평균'] = monthly_avg
            
            st.dataframe(heatmap_pivot.style.format("{:.2%}", na_rep="").background_gradient(cmap='RdYlGn', axis=None))

        # --- [수정] '전략 기여도 분석' 테이블 ---
        st.subheader("💎 개별 자산 전략 기여도 분석")
        with st.spinner('개별 자산 기여도 계산 중...'):
            # 1. 필요한 데이터 추출
            target_weights = results.get('target_weights')
            prices = results.get('prices')
            config = results.get('config')
            
            contribution_data = []
            
            # 2. 리밸런싱 주기에 맞는 기간별 수익률 계산 (안전한 reindex 사용)
            rebal_dates = target_weights.index
            periodic_prices = prices.reindex(rebal_dates, method='nearest')
            periodic_returns = periodic_prices.pct_change()

            # 3. 분석할 전체 자산 목록 준비 (중복 제거)
            aggressive_tickers = config['tickers']['AGGRESSIVE']
            defensive_tickers = config['tickers']['DEFENSIVE']
            all_assets = list(dict.fromkeys(aggressive_tickers + defensive_tickers))

            # 4. 각 자산별 기여도 계산
            for asset in all_assets:
                if asset in target_weights.columns:
                    holding_periods = target_weights.index[target_weights[asset] > 0]
                    
                    months_held = len(holding_periods)
                    if months_held == 0:
                        continue

                    returns_when_held = periodic_returns.loc[holding_periods, asset].dropna()
                    
                    avg_return = returns_when_held.mean()
                    win_rate = (returns_when_held > 0).sum() / len(returns_when_held) if not returns_when_held.empty else 0

                    # --- ▼▼▼ 전체 이름(Full Name) 찾아서 합치는 로직 ▼▼▼ ---
                    full_name = asset # 기본값은 티커로 설정
                    if etf_df is not None:
                        match = etf_df[etf_df['Ticker'] == asset]
                        if not match.empty:
                            # CSV 파일에 해당 티커 정보가 있으면 전체 이름으로 변경
                            full_name = match.iloc[0]['Name']
                    
                    # 최종적으로 표시될 이름 형식 (예: SPY - SPDR S&P 500...)
                    display_name = f"{asset} - {full_name}" if asset != full_name else asset
                    # --- ▲▲▲ 로직 끝 ▲▲▲ ---

                    contribution_data.append({
                        "자산 (Asset)": display_name, # 티커 대신 display_name 사용
                        "총 보유 횟수": f"{months_held}회",
                        "평균 보유 기간 수익률": avg_return,
                        "보유 시 승률": win_rate
                    })

            # 5. 결과 테이블 표시
            if contribution_data:
                contribution_df = pd.DataFrame(contribution_data).set_index("자산 (Asset)")
                st.dataframe(contribution_df.style.format({
                    "평균 보유 기간 수익률": "{:,.2%}",
                    "보유 시 승률": "{:,.2%}"
                }))
            else:
                st.info("기여도를 분석할 자산 데이터가 없습니다.")
                
        with st.expander("⚖️ 월별 리밸런싱 내역 보기 (전체 기간)"):
            #recent_weights = target_weights[target_weights.index > (target_weights.index.max() - pd.DateOffset(months=12))]
            #for date, weights in reversed(list(recent_weights.iterrows())):
            for date, weights in reversed(list(target_weights.iterrows())):
                holdings = weights[weights > 0]
                # 리밸런싱 판단 시점(date)을 기준으로 다음 달을 표시
                display_month_str = (date + pd.DateOffset(months=1)).strftime('%Y-%m')
    
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

                    # 각 기여금 입금 날짜에 해당하는 누적 투자 원금 계산
                    # (입금 후 수익률 적용 기준 - calculate_cumulative_returns_with_dca와 동일한 로직)
                    monthly_adds = pd.Series(0.0, index=portfolio_value.index)
                    for cd in contribution_dates:
                        if cd in monthly_adds.index:
                            monthly_adds.loc[cd] += monthly_contribution
                    
                    # 첫날: 초기 투자금 세팅
                    if not monthly_adds.empty:
                        monthly_adds.iloc[0] = initial_capital + monthly_adds.iloc[0]
                    
                    cumulative_contributions = monthly_adds.cumsum()

                    # 수익률(%) = (현재 자산 - 누적 원금) / 누적 원금
                    cumulative_return_pct = ((portfolio_value - cumulative_contributions) / cumulative_contributions.replace(0, np.nan)) * 100
                    
                    ax1.plot(cumulative_return_pct, label=result_name, linewidth=1.0)

            ax1.set_title('누적 수익률 비교', fontsize=16)
            ax1.set_xlabel('날짜'); ax1.set_ylabel('누적 수익률 (%)')
            ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y:,.0f}%'))
            ax1.legend(loc='upper left'); ax1.grid(True, which="both", ls="--", linewidth=0.5)
            st.pyplot(fig1)
            plt.close(fig1)

            st.divider()
            st.subheader("📉 하락폭(Drawdown) 비교 그래프")
            
            fig2, ax2 = plt.subplots(figsize=(10, 5))

            for result_item in selected_results_structured:
                result_name = result_item['name']
                result_data = result_item['data']

                timeseries = result_data.get('timeseries', {})
                dd_series = timeseries.get('strategy_drawdown')

                if dd_series is not None:
                    # 메인 탭과 동일하게 빈 데이터(NaN)로 인한 그래프 오류를 방지합니다.
                    dd_series = dd_series.astype(float).fillna(0)
                    
                    ax2.plot(dd_series, label=result_name, linewidth=1.0)
                    ax2.fill_between(dd_series.index, dd_series, 0, alpha=0.1) # 하락폭 영역 음영 처리

            ax2.set_title('하락폭(Drawdown) 비교', fontsize=16)
            ax2.set_xlabel('날짜'); ax2.set_ylabel('하락폭 (%)')
            ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax2.legend(loc='lower right'); ax2.grid(True, which="both", ls="--", linewidth=0.5)
            st.pyplot(fig2)
            plt.close(fig2)


            
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
