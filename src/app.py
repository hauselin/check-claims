# %% load modules

from pathlib import Path

import gensim.downloader as api
import numpy as np
import pandas as pd
import spacy
import streamlit as st
from spacytextblob.spacytextblob import SpacyTextBlob

import utils
from utils import factcheck, get_entities, tokenize, wmd

# %%

st.markdown("## Claim checker")
st.write(
    "It takes time to load the model, so be patient when the app runs the first time. Expect to wait up to 5 mins initially, but subsequent runs will run faster/immediately."
)

# %%
@st.cache(allow_output_mutation=True)
def load_model():
    # model = api.load("word2vec-google-news-300")
    # model = api.load("fasttext-wiki-news-subwords-300")
    model = api.load("glove-wiki-gigaword-50")
    model.wmdistance("hi", "hi")
    return model


model = load_model()

# %%

text = st.text_input(label="Enter some text (e.g., tweet, headline)")

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacytextblob")
doc = nlp(text)

if len(text) > 2:
    st.write(
        "Polarity: ",
        np.round(doc._.polarity, 2),
        ". Subjectivity: ",
        np.round(doc._.subjectivity),
        2,
    )

query = get_entities(
    doc, labels=["PERSON", "ORG", "GPE"], output="list", verbose=True, add=True
)
print(f"Entities: {query}")

if not query:
    query = tokenize(doc, output="list", verbose=True)
    print(f"Tokens: {query}")

if not query and text != "":
    st.write("Enter something better than this...")
    st.stop()

if query:
    st.markdown("##### Entities/tokens detected in input")
    st.write(query)
    st.write("")

# %%

claims, _ = factcheck(
    query, page_size=15, verbose=True, key=st.secrets["google_factcheck_key"]
)

if claims.shape[0] == 0:
    st.stop()

#%% check closest match

dists = []

for r in claims.itertuples():
    doc_claim = nlp(r.text)
    dist_token = wmd(
        doc, doc_claim, model, tokenizer=tokenize, tokenize_output="string"
    )
    del dist_token["tokenizer"]
    dist = dist_token
    dist["language"] = r.languageCode
    dist["rating"] = r.textualRating
    dist["factcheck_site"] = r.url
    dist["claim"] = r.text
    dist["factcheck_pub"] = r.publisher_site
    dists.append(dist)

df_dists = pd.concat(dists, ignore_index=True)

# %%

similar = (
    df_dists.sort_values(["wmd"])
    .query("language == 'en'")
    .head(15)
    .reset_index(drop=True)
)

st.markdown("### Most similar claims")
st.write("Smaller `wmd`, more similar input text is to fact-checked claim.")

cols = [
    "wmd",
    "rating",
    "factcheck_site",
    "text_tokens",
    "claim_tokens",
    "claim",
    "text_polarity",
    "claim_polarity",
    "text_subjectivity",
    "claim_subjectivity",
]
st.write(similar[cols])

#%%
