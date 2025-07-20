import yfinance as yf
from yahooquery import Ticker
import pandas as pd
import os


class building_dataset:
    def __init__(self):
        pass

    def get_static_info(self, ticker_symbol):
        t = yf.Ticker(ticker_symbol)
        info = t.info
        return {
            'Ticker': ticker_symbol,
            'Sector': info.get('sector'),
            'Industry': info.get('industry'),
            'Shares Outstanding': info.get('sharesOutstanding'),
            'Beta': info.get('beta'),
            'PE Ratio': info.get('trailingPE'),
        }

    def get_latest_eps_date(self, ticker_symbol):
        t = yf.Ticker(ticker_symbol)
        try:
            dates = t.earnings_dates.index.sort_values(ascending=False)
            return dates[0] if len(dates) > 0 else None
        except:
            return None

    def get_price_on_date(self, ticker, date):
        data = ticker.history(start=date, end=date + pd.Timedelta(days=1))
        return data['Close'].iloc[0] if not data.empty else None

    def get_marketcap_on_date(self, ticker, date, shares_outstanding):
        price = self.get_price_on_date(ticker, date)
        return price * shares_outstanding if price and shares_outstanding else None

    def get_eps_surprise(self, ticker_symbol, target_period=None):
        t = Ticker(ticker_symbol)
        df = t.earning_history
        if df is None or df.empty:
            return None
        df = df.sort_values('period', ascending=False).reset_index(drop=True)
        row = df.iloc[0] if target_period is None else df[df['period'] == target_period].iloc[0]
        row['EPS Surprise'] = row['epsActual'] - row['epsEstimate']
        return {
            'EPS Actual': row['epsActual'],
            'EPS Estimate': row['epsEstimate'],
            'EPS Surprise': row['EPS Surprise'],
            'Surprise %': row['surprisePercent']
        }

    def get_quarterly_revenue(self, ticker_symbol):
        t = yf.Ticker(ticker_symbol)
        try:
            return t.quarterly_financials.loc['Total Revenue'].iloc[0]
        except:
            return None

    def get_price_change_after_eps(self, ticker_symbol, date):
        t = yf.Ticker(ticker_symbol)
        try:
            p1 = t.history(start=date, end=date + pd.Timedelta(days=1))['Close']
            p2 = t.history(start=date + pd.Timedelta(days=1), end=date + pd.Timedelta(days=2))['Close']
            if not p1.empty and not p2.empty:
                return ((p2.iloc[0] - p1.iloc[0]) / p1.iloc[0]) * 100
        except:
            return None
        return None

    def get_pre_eps_metrics(self, ticker_symbol, eps_date):
        eps_date = pd.to_datetime(eps_date)
        start_date = eps_date - pd.Timedelta(days=30)
        end_date = eps_date - pd.Timedelta(days=1)
        t = Ticker(ticker_symbol)
        hist = t.history(start=start_date.strftime('%Y-%m-%d'), end=eps_date.strftime('%Y-%m-%d'))

        if isinstance(hist.index, pd.MultiIndex):
            hist = hist.loc[ticker_symbol]

        if hist.empty or len(hist) < 20:
            return {}

        price_col = 'adjclose' if 'adjclose' in hist.columns else 'close'
        hist['returns'] = hist[price_col].pct_change()

        def safe_pct(col, i1, i2):
            try:
                return ((col.iloc[i1] - col.iloc[i2]) / col.iloc[i2]) * 100
            except:
                return None

        try:
            vol_5d = hist['returns'].iloc[-5:].std() * (252 ** 0.5)
            return {
                'price_pct_change_1d_before': safe_pct(hist[price_col], -1, -2),
                'price_pct_change_5d_before': safe_pct(hist[price_col], -5, -6),
                'volume_pct_change_1d_before': safe_pct(hist['volume'], -1, -2),
                'volatility_5d': vol_5d,
                'ma_5': hist[price_col].rolling(5).mean().iloc[-1],
                'ma_10': hist[price_col].rolling(10).mean().iloc[-1],
                'ma_20': hist[price_col].rolling(20).mean().iloc[-1],
            }
        except:
            return {}

    def build_latest_eps_dataset(self, ticker_list):
        rows = []
        for ticker_symbol in ticker_list:
            try:
                print(f"Processing {ticker_symbol}...")
                static = self.get_static_info(ticker_symbol)
                eps_date = self.get_latest_eps_date(ticker_symbol)
                if eps_date is None:
                    continue

                row = static.copy()
                row['Earnings Date'] = eps_date.date()

                row['Market Cap'] = self.get_marketcap_on_date(yf.Ticker(ticker_symbol), eps_date, static['Shares Outstanding'])
                row['Revenue'] = self.get_quarterly_revenue(ticker_symbol)
                row['Price % Change After EPS (1d)'] = self.get_price_change_after_eps(ticker_symbol, eps_date)

                eps_info = self.get_eps_surprise(ticker_symbol)
                if eps_info:
                    row.update(eps_info)

                pre_metrics = self.get_pre_eps_metrics(ticker_symbol, eps_date)
                row.update(pre_metrics)

                rows.append(row)

            except Exception as e:
                print(f"Error with {ticker_symbol}: {e}")

        df = pd.DataFrame(rows)
       # os.makedirs("eps_pipeline", exist_ok=True)
        df.to_csv("prediction\\query_script\\ticker_ds.csv", index=False)
        return df