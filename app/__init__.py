"""Supplier-discovery demo package.

Each module maps to one stage of the find -> rank -> verify architecture:
    understand.py -> query understanding
    store.py      -> golden record store + entity resolution (data layer)
    retrieve.py   -> structured + semantic retrievers ("find")
    fuse.py       -> reciprocal rank fusion (recall)
    rank.py       -> hard-filter gate + weighted MCDA ("rank")
    verify.py     -> MOCK verification & citation ("verify")
    pipeline.py   -> orchestrates the stages and builds the Trace
    main.py       -> FastAPI app + frontend
    eval.py       -> tiny gold-set eval (precision@5 + faithfulness)
"""
