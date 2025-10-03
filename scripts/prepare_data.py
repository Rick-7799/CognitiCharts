import numpy as np, pandas as pd, os
np.random.seed(42)
dates = pd.date_range("2018-01-01", periods=600, freq="B")
drift = 0.05/252.0; vol = 1.2/np.sqrt(252)
rets = np.random.normal(drift, vol, size=len(dates))
price = 100*np.exp(np.cumsum(rets))
high = price*(1+np.random.rand(len(dates))*0.01); low = price*(1-np.random.rand(len(dates))*0.01)
openp = price*(1+np.random.randn(len(dates))*0.002); close = price
volu = np.random.randint(1000,6000,size=len(dates))
df = pd.DataFrame({"date":dates.strftime("%Y-%m-%d"),"open":openp.round(2),"high":high.round(2),
                   "low":low.round(2),"close":close.round(2),"volume":volu})
os.makedirs("sample_data", exist_ok=True); df.to_csv("sample_data/sample_prices.csv", index=False)
print("Wrote sample_data/sample_prices.csv with", len(df), "rows")
