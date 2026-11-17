"""Scenario serialisation (T65): shareable URL payload and named JSON scenarios.

Pure functions, no streamlit. Two formats:

* URL payload (``?p=``): the ``RunContext`` *diff from defaults*, as ``{"v": 1, ...fields}``, JSON -> zlib ->
  urlsafe base64 (no padding). A defaults-only context gives an empty payload.
* Scenario file: ``{"version": 1, "name", "saved_at", "context": <full dump>, "strategy_keys": [...]}``.

Decoding ignores unknown fields (a link made by a newer version still loads) and raises ``ScenarioError`` with a
user-presentable message on anything corrupted or invalid.
"""

from __future__ import annotations

import base64
import binascii
import json
import zlib
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from src.application.run_context import RunContext

PAYLOAD_VERSION = 1
SCENARIO_VERSION = 1
MAX_PAYLOAD_CHARS = 20_000  # encoded size
MAX_DECODED_BYTES = 200_000  # guard against decompression bombs
# Fields that describe the data, not a user choice: never part of a link.
EXCLUDED_FIELDS = frozenset({"archetypes_version"})


class ScenarioError(ValueError):
    """A payload or scenario file that cannot be read. The message is shown to the user as is."""


def _known_fields() -> set[str]:
    return set(RunContext.model_fields) - EXCLUDED_FIELDS


def _context_from_fields(fields: dict[str, Any]) -> RunContext:
    known = _known_fields()
    clean = {k: v for k, v in fields.items() if k in known}
    try:
        return RunContext.model_validate(clean)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "?"
        raise ScenarioError(f"Valeur invalide pour « {where} » : {first['msg']}.") from exc


def context_from_json(text: str | bytes) -> RunContext:
    """Validate a stored full-context JSON document (unknown fields ignored). Raises ``ScenarioError``."""
    try:
        data = json.loads(text)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ScenarioError("Scénario enregistré illisible ou corrompu.") from exc
    if not isinstance(data, dict):
        raise ScenarioError("Scénario enregistré illisible ou corrompu.")
    return _context_from_fields(data)


def context_diff(ctx: RunContext) -> dict[str, Any]:
    """JSON-compatible dict of the fields of ``ctx`` that differ from the defaults."""
    default = RunContext().model_dump(mode="json")
    current = ctx.model_dump(mode="json")
    return {k: v for k, v in current.items() if k in _known_fields() and v != default[k]}


def encode_context(ctx: RunContext) -> str:
    """Compact URL-safe payload for ``ctx``; empty string when everything is at its default."""
    diff = context_diff(ctx)
    if not diff:
        return ""
    raw = json.dumps({"v": PAYLOAD_VERSION, **diff}, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    packed = zlib.compress(raw.encode("utf-8"), 9)
    return base64.urlsafe_b64encode(packed).rstrip(b"=").decode("ascii")


def decode_context(payload: str) -> RunContext:
    """Inverse of :func:`encode_context`. Empty payload = defaults. Raises ``ScenarioError``."""
    payload = (payload or "").strip()
    if not payload:
        return RunContext()
    if len(payload) > MAX_PAYLOAD_CHARS:
        raise ScenarioError("Lien de scénario trop long.")
    try:
        packed = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        decomp = zlib.decompressobj()
        raw = decomp.decompress(packed, MAX_DECODED_BYTES)
        if decomp.unconsumed_tail:
            raise ScenarioError("Lien de scénario trop volumineux.")
        if not decomp.eof:  # truncated stream
            raise ScenarioError("Lien de scénario illisible ou corrompu.")
        data = json.loads(raw.decode("utf-8"))
    except ScenarioError:
        raise
    except (binascii.Error, zlib.error, UnicodeDecodeError, ValueError) as exc:  # JSONDecodeError is a ValueError
        raise ScenarioError("Lien de scénario illisible ou corrompu.") from exc
    if not isinstance(data, dict):
        raise ScenarioError("Lien de scénario illisible ou corrompu.")
    version = data.pop("v", None)
    if version != PAYLOAD_VERSION:
        raise ScenarioError(f"Version de lien non prise en charge ({version!r}).")
    return _context_from_fields(data)


def build_query_params(ctx: RunContext, strategy_idx: int | None = None) -> dict[str, str]:
    """Query parameters of a shareable link (``p`` omitted when all defaults, ``strategy`` when None)."""
    params: dict[str, str] = {}
    payload = encode_context(ctx)
    if payload:
        params["p"] = payload
    if strategy_idx is not None:
        params["strategy"] = str(strategy_idx)
    return params


# ---------------------------------------------------------------------------
# Scenario files
# ---------------------------------------------------------------------------
def scenario_to_json(
    name: str,
    ctx: RunContext,
    strategy_keys: list[str] | None = None,
    saved_at: datetime | None = None,
) -> str:
    """JSON text of a named scenario (full context dump, optional keys of the top strategies)."""
    stamp = (saved_at or datetime.now()).isoformat(timespec="seconds")
    doc = {
        "version": SCENARIO_VERSION,
        "name": name.strip() or "Scénario",
        "saved_at": stamp,
        "context": ctx.model_dump(mode="json"),
        "strategy_keys": list(strategy_keys or []),
    }
    return json.dumps(doc, indent=2, ensure_ascii=False)


def scenario_from_json(text: str | bytes) -> tuple[str, RunContext, list[str]]:
    """Parse a scenario file into ``(name, context, strategy_keys)``. Raises ``ScenarioError``."""
    try:
        doc = json.loads(text)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ScenarioError("Ce fichier n'est pas un JSON valide.") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("context"), dict):
        raise ScenarioError("Ce fichier n'est pas un scénario (clé « context » absente).")
    version = doc.get("version")
    if version != SCENARIO_VERSION:
        raise ScenarioError(f"Version de scénario non prise en charge ({version!r}).")
    ctx = _context_from_fields(doc["context"])
    name = str(doc.get("name") or "Scénario")
    keys = doc.get("strategy_keys")
    strategy_keys = [str(k) for k in keys] if isinstance(keys, list) else []
    return name, ctx, strategy_keys
