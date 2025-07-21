"""Init constructor for package
"""

from pathlib import Path

__pkgname__ = 'hexrec'

HEXREC_ROOT = Path(__file__).parent
HEXREC_BASE = HEXREC_ROOT.parent
HEXREC_BIN = HEXREC_ROOT / 'bin'

HEXREC_DATA = Path().home() / 'hexrecdata'
if not Path.exists(HEXREC_DATA):
    Path.mkdir(HEXREC_DATA)

HEXREC_MODELS = HEXREC_DATA / 'models'
if not Path.exists(HEXREC_MODELS):
    Path.mkdir(HEXREC_MODELS)
