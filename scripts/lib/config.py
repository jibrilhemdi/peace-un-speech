"""Config loading and repo-root resolution."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"

_FLAGS = {
    "IGNORECASE": re.IGNORECASE,
    "VERBOSE": re.VERBOSE,
    "DOTALL": re.DOTALL,
    "MULTILINE": re.MULTILINE,
}


class ConfigError(RuntimeError):
    pass


def load_dotenv(path: Path | None = None) -> None:
    """Minimal KEY=VALUE reader for the repo-root .env.

    Does not override variables already in the environment, so an exported
    OPENROUTER_API_KEY always beats the file.
    """
    if os.environ.get("UN_SPEECHES_IGNORE_DOTENV"):
        # Escape hatch for CI and tests: guarantees the key comes from the real environment
        # and that a stray .env on the machine cannot silently supply credentials.
        return
    p = path or (REPO_ROOT / ".env")
    if not p.exists():
        return
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value



def model_tag(model_id: str | None) -> str:
    """Short, stable slug for a provider model id, for use as a data value.

    ``poolside/laguna-s-2.1:free`` -> ``laguna-s-2.1``
    ``inclusionai/ling-3.0-flash:free`` -> ``ling-3.0-flash``

    The full id stays in the ``model`` column; this is the compact form to group or dummy on.
    The vendor prefix and the ``:free`` suffix are dropped because neither identifies the
    instrument — the same model served free and paid produces the same labels.
    """
    if not model_id:
        return "unknown"
    tag = str(model_id).split("/")[-1]
    for suffix in (":free", ":beta", ":extended", ":nitro"):
        if tag.endswith(suffix):
            tag = tag[: -len(suffix)]
    return tag or "unknown"

@dataclass(frozen=True)
class ProceduralRule:
    """One paragraph-level procedural rule, compiled from config.yaml."""

    name: str
    kind: str | None
    pattern: re.Pattern[str] | None
    require: re.Pattern[str] | None
    mode: str
    max_words: int | None

    def fires(self, paragraph: str, prev_procedural: bool) -> bool:
        if self.kind == "inherit_previous_quote":
            return prev_procedural and paragraph.lstrip().startswith(("“", '"', "‘", "'"))
        if self.pattern is None:
            return False
        hit = (
            self.pattern.match(paragraph)
            if self.mode == "match"
            else self.pattern.search(paragraph)
        )
        if not hit:
            return False
        if self.max_words is not None and len(paragraph.split()) > self.max_words:
            return False
        if self.require is not None and not self.require.search(paragraph):
            return False
        return True


class Config:
    """Thin typed wrapper over config.yaml. Paths are resolved against the repo root."""

    def __init__(self, raw: dict[str, Any], path: Path):
        self.raw = raw
        self.path = path

    # -- loading ------------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        p = Path(path) if path else CONFIG_PATH
        if not p.exists():
            raise ConfigError(f"config not found: {p}")
        with p.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ConfigError(f"config is not a mapping: {p}")
        return cls(raw, p)

    def section(self, name: str) -> dict[str, Any]:
        v = self.raw.get(name)
        if not isinstance(v, dict):
            raise ConfigError(f"config section '{name}' missing or not a mapping in {self.path}")
        return v

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    # -- paths --------------------------------------------------------------------------
    def path_for(self, key: str) -> Path:
        paths = self.section("paths")
        if key not in paths:
            raise ConfigError(f"paths.{key} not set in {self.path}")
        return (REPO_ROOT / str(paths[key])).resolve()

    # -- segmentation -------------------------------------------------------------------
    def procedural_rules(self) -> list[ProceduralRule]:
        out: list[ProceduralRule] = []
        for i, r in enumerate(self.get("segmentation.procedural_rules", []) or []):
            if not isinstance(r, dict):
                raise ConfigError(f"segmentation.procedural_rules[{i}] is not a mapping")
            name = str(r.get("name", f"rule_{i}"))
            kind = r.get("kind")
            pattern = None
            require = None
            if kind is None:
                if "pattern" not in r:
                    raise ConfigError(f"procedural rule '{name}' has neither 'pattern' nor 'kind'")
                flags = 0
                for f in r.get("flags", ["IGNORECASE"]):
                    if f not in _FLAGS:
                        raise ConfigError(f"procedural rule '{name}': unknown regex flag {f!r}")
                    flags |= _FLAGS[f]
                try:
                    pattern = re.compile(str(r["pattern"]), flags)
                except re.error as exc:
                    raise ConfigError(f"procedural rule '{name}': bad regex — {exc}") from exc
                if r.get("require"):
                    try:
                        require = re.compile(str(r["require"]), flags or re.IGNORECASE)
                    except re.error as exc:
                        raise ConfigError(
                            f"procedural rule '{name}': bad 'require' regex — {exc}"
                        ) from exc
            elif kind != "inherit_previous_quote":
                raise ConfigError(f"procedural rule '{name}': unknown kind {kind!r}")
            out.append(
                ProceduralRule(
                    name=name,
                    kind=kind,
                    pattern=pattern,
                    require=require,
                    mode=str(r.get("mode", "match")),
                    max_words=int(r["max_words"]) if r.get("max_words") is not None else None,
                )
            )
        return out

    def bloc_map(self) -> dict[str, str]:
        name = self.get("bloc_map_name")
        maps = self.get("bloc_maps", {}) or {}
        if name not in maps:
            raise ConfigError(f"bloc_map_name '{name}' not found in bloc_maps")
        return {str(k): str(v) for k, v in maps[name].items()}

    # -- provider -----------------------------------------------------------------------
    @staticmethod
    def api_key() -> str:
        """Read the key from the environment, falling back to a .env file at the repo root.

        .env is gitignored and is never read into config, the database, or any log — it is
        loaded into os.environ and used for the Authorization header only. A real environment
        variable always wins, so `export OPENROUTER_API_KEY=...` overrides the file.
        """
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            load_dotenv()
            key = os.environ.get("OPENROUTER_API_KEY", "").strip()

        if not key or key.startswith("PASTE_YOUR_KEY"):
            raise ConfigError(
                ("OPENROUTER_API_KEY is still the placeholder in .env."
                 if key else "OPENROUTER_API_KEY is not set.")
                + "\n\nPut your key in the .env file at the repo root:\n"
                f"    {REPO_ROOT / '.env'}\n"
                "    OPENROUTER_API_KEY=sk-or-v1-...\n\n"
                "Or export it in your shell instead:\n"
                "    export OPENROUTER_API_KEY=sk-or-v1-...\n\n"
                ".env is gitignored. The key is never written to config.yaml, the database, "
                "an export, or a log."
            )
        return key

    def models(self, override: str | None = None) -> list[str]:
        """Primary model first, then configured fallbacks."""
        primary = override or self.get("provider.model")
        if not primary:
            raise ConfigError("provider.model is not set in config.yaml")
        chain = [str(primary)]
        for m in self.get("provider.fallback_models", []) or []:
            if str(m) != chain[0]:
                chain.append(str(m))
        return chain
