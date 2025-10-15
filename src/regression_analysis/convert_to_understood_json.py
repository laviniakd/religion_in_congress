import pandas as pd
import json

DATA_FILE = "/data/laviniad/congress_errata/congress_df_limited.json"
OUT_FILE = "/data/laviniad/congress_errata/cdf_limited_readable.json"
data = pd.read_json(DATA_FILE)
data = data.drop('text', axis=1)

with open(OUT_FILE, 'w') as f:
    f.write(data.to_json(orient='records', lines=True))
