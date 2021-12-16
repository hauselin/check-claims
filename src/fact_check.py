# %% load modules

from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import gensim.downloader as api

import utils
from utils import get_entities, tokenize, factcheck, wmd

pd.set_option(
    "display.max_rows",
    8,
    "display.max_columns",
    None,
    "display.width",
    None,
    "display.expand_frame_repr",
    True,
    "display.max_colwidth",
    None,
)

np.set_printoptions(
    edgeitems=5,
    linewidth=233,
    precision=4,
    sign=" ",
    suppress=True,
    threshold=50,
    formatter=None,
)

# model = api.load("word2vec-google-news-300")

#%%

text = "So tonight, a sheriff in #Michigan suggested it might be appropriate to do a citizen's arrest of the Governor, and Keith Olberman suggested Amy Coney Barrett should be prosecuted and removed from society. So, yes, national unity seems just around the corner."

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacytextblob")
doc = nlp(text)

query = get_entities(doc, labels=["PERSON", "ORG", "GPE"], output="list", verbose=True)
print(f"Entities: {query}")

#%% find related claims

claims, _ = factcheck(query, page_size=15, verbose=True)

# save matching claims (so no need to search for already-search queries)
# claims_df.to_csv(f"../data/{query}.csv", index=False)


#%% check closest match

dists = []

for r in claims.itertuples():
    doc_claim = nlp(r.text)
    dist_token = wmd(
        doc, doc_claim, model, tokenizer=tokenize, tokenize_output="string"
    )
    dist = dist_token
    dist["language"] = r.languageCode
    dist["rating"] = r.textualRating
    dist["factcheck_site"] = r.url
    dist["factcheck_pub"] = r.publisher_site
    dists.append(dist)

df_dists = pd.concat(dists, ignore_index=True)

# %%

similar = (
    df_dists.sort_values(["wmd"])
    .query("language == 'en'")
    .head(5)
    .reset_index(drop=True)
)

similar
cols = ["wmd", "rating", "claim_tokens", "factcheck_site"]
print(text)
similar[cols]

#%%
