# %% load modules

from pathlib import Path
import numpy as np
import pandas as pd

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

#%%

dist_token = wmd(doc, doc_claim, model, tokenizer=tokenize, tokenize_output="string")
dist_entity = wmd(
    doc, doc_claim, model, tokenizer=get_entities, tokenize_output="string"
)

dist = pd.concat([dist_token, dist_entity], ignore_index=True)
