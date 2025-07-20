import pandas as pd
import sqlite3


df = pd.read_csv(r'data\cleaned_eps_data.csv')  
ignore_c = ['Ticker', 'Sector', 'Industry', 'Price % Change After EPS (1d)', 'Earnings Date']

df = df.drop(columns=ignore_c, errors='ignore')

conn = sqlite3.connect(r'data\initial_db.db')
df.to_sql('eps_data', conn, if_exists='replace', index=False)


conn.close()
print("imported")
