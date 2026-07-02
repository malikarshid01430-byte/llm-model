import os
import tempfile
from pathlib import Path

_test_db = Path(tempfile.gettempdir()) / f"edullm-pytest-{os.getpid()}.db"
if _test_db.exists():
    _test_db.unlink()

os.environ["POSTGRES_URL"] = f"sqlite+aiosqlite:///{_test_db}"
os.environ["DEBUG"] = "true"
