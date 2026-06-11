"""
Backend.recommender package initializer
"""

import importlib.util
import os
import sys


# Load Backend/recommender.py
_file = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "recommender.py"
)

_spec = importlib.util.spec_from_file_location(
    "Backend._recommender_impl",
    _file
)

_mod = importlib.util.module_from_spec(_spec)

sys.modules[_spec.name] = _mod

_spec.loader.exec_module(_mod)


# Export functions

recommend_for_user = _mod.recommend_for_user

get_sample_users = _mod.get_sample_users

get_item_catalogue = _mod.get_item_catalogue



# Export model loader if exists

if hasattr(_mod, "_load_neural_model"):
    _load_neural_model = _mod._load_neural_model



# Export analytics variables safely

if hasattr(_mod, "_POP"):
    _POP = _mod._POP
else:
    _POP = {}


if hasattr(_mod, "_CONTEXT"):
    _CONTEXT = _mod._CONTEXT
else:
    import pandas as pd
    _CONTEXT = pd.DataFrame()



# FAISS

try:
    from .faiss_index import search
except Exception:
    pass