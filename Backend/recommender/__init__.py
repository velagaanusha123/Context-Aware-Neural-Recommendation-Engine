"""
Backend.recommender  –  package initialiser
--------------------------------------------
There is a naming collision: both  Backend/recommender.py  (module file)
and  Backend/recommender/  (this package directory) exist.  Python picks
the package, so we must explicitly load the .py file and re‑export every
symbol that  Backend/main.py  expects.
"""

import importlib.util, os, sys

# ── Load Backend/recommender.py by absolute file path ──────────────────────
_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "recommender.py")
_spec = importlib.util.spec_from_file_location("Backend._recommender_impl", _file)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod          # cache so it isn't loaded twice
_spec.loader.exec_module(_mod)

# ── Re‑export public API used by Backend/main.py ──────────────────────────
recommend_for_user = _mod.recommend_for_user
get_sample_users   = _mod.get_sample_users
get_item_catalogue = _mod.get_item_catalogue
_load_neural_model = _mod._load_neural_model

# ── Re‑export internal variables used by analytics endpoints ──────────────
_CONTEXT   = _mod._CONTEXT
_ITEM_META = _mod._ITEM_META

# ── Also expose FAISS helper from the sub‑module ──────────────────────────
from .faiss_index import search   # noqa: F401
