# -*- coding: utf-8 -*-
"""GitHub Actions 에서 실행: 브라우저 버전(site/)이 읽을 수정종가(Adj Close) 파일 site/prices.csv 생성.

대상 티커 = Stock_list.csv + 앱 기본값에 쓰이는 티커.
이번에 수집 실패한 티커는 직전 배포본(prices.csv)의 데이터를 그대로 유지한다.
"""
import io
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ROOT  = Path(__file__).resolve().parent.parent
SITE  = ROOT / 'site'
START = '1990-01-01'

# Quantest_v10.py 의 기본값(텍스트 입력 기본 자산군, A-Core 대체자산 힌트, 벤치마크)
DEFAULT_TICKERS = [
    '^GSPC',   # A-Core 시장 국면 판단용 S&P500 지수
    'SPY', 'IWM', 'VEA', 'VWO', 'VNQ', 'DBC', 'IEF', 'TLT', 'BIL', 'TIP',
    'GLD', 'IAU', 'SLV', 'USO', 'PDBC', 'GNR', 'IYR', 'REMX',
    '411060.KS', '476760.KS', '352560.KS', '305080.KS', '276000.KS', '415920.KS',
]


def load_previous() -> pd.DataFrame:
    url = os.environ.get('PAGES_URL', '').rstrip('/')
    if not url:
        return pd.DataFrame()
    try:
        r = requests.get(f'{url}/prices.csv', timeout=60)
        if r.ok:
            return pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
    except Exception as e:
        print(f'직전 데이터 로드 실패: {e}')
    return pd.DataFrame()


def main():
    stock_list = pd.read_csv(ROOT / 'Stock_list.csv', encoding='utf-8')
    tickers = list(dict.fromkeys(
        [str(t).strip().upper() for t in stock_list['Ticker'].dropna() if str(t).strip()] + DEFAULT_TICKERS
    ))
    print(f'대상 {len(tickers)}개: {", ".join(tickers)}')

    raw = yf.download(tickers, start=START, progress=False, auto_adjust=False, threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        adj = raw['Adj Close'] if 'Adj Close' in raw.columns.get_level_values(0) else pd.DataFrame()
        close = raw['Close']
    else:   # 티커 1개
        adj = raw[['Adj Close']].rename(columns={'Adj Close': tickers[0]}) if 'Adj Close' in raw else pd.DataFrame()
        close = raw[['Close']].rename(columns={'Close': tickers[0]})

    prices = pd.DataFrame(index=raw.index)
    failed = []
    for t in tickers:
        s = adj[t] if t in adj.columns and adj[t].notna().any() else (close[t] if t in close.columns else None)
        if s is None or s.notna().sum() == 0:
            failed.append(t)
            continue
        prices[t] = s

    prev = load_previous()
    kept = []
    for t in failed:
        if t in prev.columns:
            prices = prices.join(prev[[t]], how='outer')
            kept.append(t)

    prices = prices.dropna(how='all').sort_index()
    prices.index.name = 'Date'
    SITE.mkdir(exist_ok=True)
    prices.to_csv(SITE / 'prices.csv', float_format='%.10g', date_format='%Y-%m-%d')

    size = (SITE / 'prices.csv').stat().st_size / 1e6
    print(f'완료: {prices.shape[1]}개 티커, {prices.index[0].date()} ~ {prices.index[-1].date()}, {size:.1f}MB')
    if failed:
        print(f'수집 실패: {", ".join(failed)} (직전 데이터 유지: {", ".join(kept) or "없음"})')
    if prices.shape[1] == 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
