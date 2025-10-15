import pandas as pd

PATH = "/home/laviniad/"
SHINGLE_FILE = PATH + "results_shingles.csv"
EMBEDDING_FILE = PATH + "results_embeddings.csv"

# load csvs
df_shingles = pd.read_csv(SHINGLE_FILE)
df_embeddings = pd.read_csv(EMBEDDING_FILE)

# concat dfs; if the same congress_idx + text combo exists in both, keep the embedding one
df = pd.concat([df_shingles, df_embeddings]).drop_duplicates(
    subset=["congress_idx", "text"], keep="last"
).reset_index(drop=True)

# dump to ~
df.to_csv(PATH + "results_combined.csv", index=False)
print("Dumped to ", PATH + "results_combined.csv")
