from __future__ import annotations

import sys
from pathlib import Path


# 让测试直接从 `src/` 导入包，而不依赖是否已经执行 `pip install -e .`。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
