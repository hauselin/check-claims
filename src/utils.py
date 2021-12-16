# %% load modules

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

try:
    from auth import google_factcheck_key
except:
    google_factcheck_key = st.secrets["google_factcheck_key"]
import requests
import urllib.parse


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

# %%


def factcheck_query(query, page_size=20, key=google_factcheck_key):
    # https://developers.google.com/fact-check/tools/api/reference/rest/v1alpha1/claims/search

    query_encode = urllib.parse.quote(query)
    # print(query_encode)
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query_encode}&pageSize={page_size}&key={key}"

    response = requests.get(url)
    if response:
        dat = response.json()
        if "claims" in dat:
            print(f"Claims found: {len(dat['claims'])}")
            claims = dat["claims"]
            try:
                next_page_token = dat["nextPageToken"]
            except:
                next_page_token = None
            return claims, next_page_token
        else:
            print("No claims found")
            return [], None
    print("Error")
    return [], None


def parse_claim(claim):
    claim_review = pd.DataFrame(claim["claimReview"])
    claim_review["text"] = claim["text"]
    claim_review
    try:
        claim_review["publisher_name"] = claim["claimReview"][0]["publisher"][
            "name"
        ].lower()
    except:
        claim_review["publisher_name"] = ""
    try:
        claim_review["publisher_site"] = claim["claimReview"][0]["publisher"][
            "site"
        ].lower()
    except:
        claim_review["publisher_site"] = ""
    del claim_review["publisher"]

    return claim_review


def factcheck(query_list, page_size=20, verbose=False, key=google_factcheck_key):
    string = " ".join(query_list)
    if verbose:
        print(f"Factchecking {string}")
    all_claims, next_page_token = factcheck_query(string, page_size=page_size, key=key)
    if all_claims:
        return (
            pd.concat([parse_claim(claim) for claim in all_claims])
            .reset_index(drop=True)
            .assign(query=string)
        ), next_page_token
    if len(query_list) > 1:
        all_claims = []
        for token in query_list:
            if verbose:
                print(f"Factchecking {token}")
            claims, next_page_token = factcheck_query(token, page_size=page_size)
            if claims is not None:
                for claim in claims:
                    all_claims.append(parse_claim(claim).assign(query=token))
    if all_claims:
        return pd.concat(all_claims).reset_index(drop=True), next_page_token

    return pd.DataFrame(), next_page_token


#%%

stopwords = {"a", "is", "are", "was", "were", "an", "am", "the", "to", "there"}


def remove_stopwords(text):
    # https://stackabuse.com/removing-stop-words-from-strings-in-python/
    text = text.lower().split()
    tokens = [token for token in text if token not in stopwords]
    return " ".join(tokens)


def tokenize(doc, output="string", verbose=False, unique=True):
    tokens = []
    for token in doc:
        if verbose:
            print(
                token.text,
                token.lemma_,
                token.pos_,
                token.tag_,
                token.dep_,
                token.is_alpha,
                token.ent_type_,
            )
        # words to skip
        if token.pos_ in [
            "PUNCT",
            "SCONJ",
            "PRON",
            "AUX",
            "ADP",
            "CCONJ",
            "DET",
            "PART",
        ]:
            continue
        if not token.lemma_.isalnum():
            continue
        lemma = token.lemma_.lower()
        if lemma in ["be", "so"]:
            continue
        if unique and lemma not in tokens or not unique:
            tokens.append(lemma)
    if output == "string":
        return " ".join(tokens)
    return tokens


def get_entities(
    doc, output="string", unique=True, verbose=False, labels=None, add=True
):
    entities = []
    for ent in doc.ents:
        if verbose:
            print(ent.text, ent.label_)
        if ent.label_ == "CARDINAL":
            continue
        if labels and ent.label_ not in labels:
            continue
        lemma = ent.lemma_.lower()
        if unique and lemma not in entities or not unique:
            entities.append(lemma)
    if len(entities) < 3:
        for token in doc:
            if token.pos_ in ["NOUN", "VERB", "ADJ", "PART"]:
                noun = token.lemma_.lower()
                if unique and noun not in entities or not unique:
                    entities.append(noun)
    if output == "string":
        return " ".join(entities)
    return entities


#%% distance

# https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html#sphx-glr-auto-examples-tutorials-run-wmd-py
def wmd(
    text_doc,
    claim_doc,
    model,
    tokenizer=tokenize,
    tokenize_output="string",
    verbose=False,
):
    output = {"tokenizer": tokenizer, "text": text_doc.text}
    # output = {"text": text_doc.text}

    text_tokens = tokenizer(text_doc, output=tokenize_output)
    output["text_tokens"] = [text_tokens]
    output["text_polarity"] = [text_doc._.polarity]
    output["text_subjectivity"] = [text_doc._.subjectivity]
    output["text_assessments"] = [text_doc._.assessments]

    claim_tokens = tokenizer(claim_doc, output=tokenize_output)
    output["claim_tokens"] = [claim_tokens]
    output["claim_polarity"] = [claim_doc._.polarity]
    output["claim_subjectivity"] = [claim_doc._.subjectivity]
    output["claim_assessments"] = [claim_doc._.assessments]

    dist = model.wmdistance(text_tokens, claim_tokens)
    output["wmd"] = dist
    if verbose:
        print(f"wmdistance: {dist}")

    return pd.DataFrame(output)
