import pandas as pd

FILE = '/data/laviniad/congress_errata/congress_df_limited.json'
OUT_FILE = '/data/laviniad/congress_errata/congress_df_limited_no_text.json'
df = pd.read_json(FILE)

df = df.drop('text', axis=1)
df.to_json(OUT_FILE)
