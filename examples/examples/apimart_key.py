import os
import re
from pathlib import Path


def get_apimart_api_key() -> str:
    """Get APIMart API key for local testing.

    Priority:
    1) Config file: api_keys.json (APIMART_API_KEY)
    2) Env: APIMART_API_KEY
    3) Env file pointer: APIMART_API_KEY_FILE (first non-empty line)
    4) Workspace root files: api key.txt / api_key.txt / .apimart_api_key

    Returns empty string if not found.
    """
    # 优先从配置文件读取
    try:
        from config_loader import get_apimart_api_key as get_from_config
        key = get_from_config()
        if key:
            return key
    except (ImportError, FileNotFoundError, ValueError, KeyError):
        pass

    key = os.getenv("APIMART_API_KEY", "").strip()
    if key:
        return key

    key_file = os.getenv("APIMART_API_KEY_FILE", "").strip()
    candidates: list[Path] = []
    if key_file:
        candidates.append(Path(key_file))

    repo_root = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            repo_root / "api key.txt",
            repo_root / "api_key.txt",
            repo_root / ".apimart_api_key",
        ]
    )

    def _normalize_line(line: str) -> str:
        line = line.strip()
        if not line or line.startswith("#"):
            return ""

        # allow formats like: Authorization: Bearer xxx
        if line.lower().startswith("authorization:"):
            line = line.split(":", 1)[1].strip()
        if line.lower().startswith("bearer "):
            line = line[7:].strip()

        # allow formats like: APIMART_API_KEY=xxx or export APIMART_API_KEY=xxx
        if "=" in line:
            maybe = line
            if maybe.lower().startswith("export "):
                maybe = maybe[7:].strip()
            k, v = maybe.split("=", 1)
            if k.strip() in {"APIMART_API_KEY", "OAI_API_KEY"}:
                line = v.strip()

        # strip common quotes
        if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
            line = line[1:-1].strip()

        return line

    def _is_ascii(s: str) -> bool:
        try:
            s.encode("ascii")
            return True
        except Exception:
            return False

    for path in candidates:
        try:
            if not path.exists() or not path.is_file():
                continue
            # utf-8-sig strips BOM if present
            text = path.read_text(encoding="utf-8-sig").strip()
            if not text:
                continue
            # allow comments/extra lines; take first non-empty non-comment line
            for line in text.splitlines():
                norm = _normalize_line(line)
                if not norm:
                    continue
                # basic sanity: token must be header-safe ASCII
                if not _is_ascii(norm):
                    # Return empty so caller can fallback/raise; avoid printing secret.
                    return ""
                # optional: accept common prefixes like sk-..., but don't enforce strict pattern
                if re.search(r"\s", norm):
                    # whitespace inside token usually indicates a copy/paste mistake
                    return ""
                return norm
        except Exception:
            continue

    return ""
