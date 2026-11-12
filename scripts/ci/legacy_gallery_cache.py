"""One-time cache bridge for the verified, unchanged legacy renderer extraction.

The mapping is exact: changes to legacy rendering, analysis, packing, libraries,
source tables or curation prevent reuse. Future bundles carry legacy_code directly.
Baseline: 177b3517cda75ee8414f6119bbe00a24338a4b24.
"""
PRE_SPLIT_GENERATORS = {
    'fc6d770fe31552c2e2f9fdfeef7b0749c509eb227796acb4556f2c06d22ab6e0': '9d4228c0b0d6c4ef1f08ec56b0b814ec58392f8eb0dc61d9b4ada6ca8b2aaadd',
    # Main 351d0dc includes trace/1 metadata. Both extracted functions have exact
    # AST parity after renaming ROOT/FIT_OPTIONS to their explicit parameters.
    '1b13af0241f0788f6d718101db5c8ef1a719efe51886f76fafdff77451909024': '7af35b636c6eb73d74ef7c607c7f4f30f55c3701c764d1288b8b0d6be5efba55',
}
