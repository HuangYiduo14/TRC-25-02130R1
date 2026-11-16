"""UTF-8 rendering and explicit text-policy validation for handoff payloads."""

from __future__ import annotations

import codecs
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


_POLICY_CLASSES = {
    "template_control",
    "command_log",
    "code_snapshot",
    "diff_or_patch",
    "plain_document",
    "binary_or_external",
}
_TOKEN = re.compile(r"\{\{[A-Za-z_][A-Za-z0-9_]*\}\}")
_POWERSHELL_VARIABLE = re.compile(r"(?<!`)\$[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True, slots=True)
class PartCRev1TextPolicy:
    relative_path: str
    policy_class: str
    encoding: str | None = "utf-8"
    template_rendered: bool = False
    forbid_powershell_variables: bool = False
    format: str | None = None
    disposition: str | None = None
    size_bytes: int | None = None
    signature: str | None = None
    reason: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PartCRev1TextPolicy":
        try:
            policy = cls(
                relative_path=str(value["relative_path"]),
                policy_class=str(value["policy_class"]),
                encoding=None if value.get("encoding") is None else str(value.get("encoding")),
                template_rendered=bool(value.get("template_rendered", False)),
                forbid_powershell_variables=bool(value.get("forbid_powershell_variables", False)),
                format=None if value.get("format") is None else str(value.get("format")),
                disposition=None if value.get("disposition") is None else str(value.get("disposition")),
                size_bytes=None if value.get("size_bytes") is None else int(value.get("size_bytes")),
                signature=None if value.get("signature") is None else str(value.get("signature")),
                reason=None if value.get("reason") is None else str(value.get("reason")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed handoff text-policy entry: {exc}") from exc
        policy.validate_contract()
        return policy

    def validate_contract(self) -> None:
        relative = PurePosixPath(self.relative_path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"Unsafe classified relative path: {self.relative_path!r}")
        if self.policy_class not in _POLICY_CLASSES:
            raise ValueError(f"Unknown text-policy class: {self.policy_class!r}")
        if self.policy_class == "binary_or_external":
            if not self.format or not self.disposition or not self.reason:
                raise ValueError("binary_or_external entries require format, disposition, and reason")
            if self.disposition not in {"binary_included", "external_only"}:
                raise ValueError(f"Unsupported binary/external disposition: {self.disposition!r}")
            if self.size_bytes is None or self.size_bytes < 0 or not self.signature:
                raise ValueError("binary_or_external entries require nonnegative size_bytes and signature")
        elif self.encoding != "utf-8":
            raise ValueError("All classified handoff text must declare encoding='utf-8'")


def _forbidden_control_positions(text: str) -> list[int]:
    return [
        index
        for index, character in enumerate(text)
        if ((ord(character) < 32 and character not in "\t\n\r") or 0x7F <= ord(character) <= 0x9F)
    ]


def partC_rev1_render_handoff_template(template: str, variables: Mapping[str, Any]) -> str:
    rendered = str(template)
    for name, value in variables.items():
        rendered = rendered.replace("{{" + str(name) + "}}", str(value))
    unresolved = sorted(set(_TOKEN.findall(rendered)))
    if unresolved:
        raise ValueError(f"Unresolved handoff template tokens: {unresolved}")
    if _forbidden_control_positions(rendered):
        raise ValueError("Rendered handoff template contains forbidden control characters")
    return rendered


def partC_rev1_write_utf8_text(path: str | Path, text: str) -> Path:
    destination = Path(path)
    if _forbidden_control_positions(text):
        raise ValueError(f"Refusing to write forbidden control characters: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8", errors="strict", newline="\n")
    return destination


def partC_rev1_convert_text_to_utf8(source: str | Path, destination: str | Path) -> Path:
    raw = Path(source).read_bytes()
    if raw.startswith(codecs.BOM_UTF16_LE):
        text = raw[len(codecs.BOM_UTF16_LE) :].decode("utf-16-le", errors="strict")
    elif raw.startswith(codecs.BOM_UTF16_BE):
        text = raw[len(codecs.BOM_UTF16_BE) :].decode("utf-16-be", errors="strict")
    elif raw.startswith(codecs.BOM_UTF8):
        text = raw[len(codecs.BOM_UTF8) :].decode("utf-8", errors="strict")
    else:
        text = raw.decode("utf-8", errors="strict")
    return partC_rev1_write_utf8_text(destination, text)


def _load_classification_manifest(value: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[PartCRev1TextPolicy]:
    if isinstance(value, (str, Path)):
        payload = json.loads(Path(value).read_text(encoding="utf-8"))
    else:
        payload = value
    if isinstance(payload, Mapping):
        if payload.get("schema_version") != "1.0":
            raise ValueError(f"Unsupported classification schema version: {payload.get('schema_version')!r}")
        entries = payload.get("files")
    else:
        entries = payload
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
        raise ValueError("Classification manifest must contain a files sequence")
    policies = [PartCRev1TextPolicy.from_mapping(item) for item in entries]
    paths = [item.relative_path for item in policies]
    if len(paths) != len(set(paths)):
        raise ValueError("Classification manifest contains duplicate paths")
    return policies


def _validate_text(path: Path, policy: PartCRev1TextPolicy) -> dict[str, Any]:
    raw = path.read_bytes()
    if raw.startswith(codecs.BOM_UTF8) or raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        raise ValueError(f"Classified text must be UTF-8 without BOM: {policy.relative_path}")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 in {policy.relative_path}: {exc}") from exc
    positions = _forbidden_control_positions(text)
    if positions:
        raise ValueError(f"Forbidden control characters in {policy.relative_path} at {positions[:10]}")
    reject_tokens = policy.policy_class == "template_control" or (
        policy.policy_class == "plain_document" and policy.template_rendered
    )
    if reject_tokens:
        tokens = sorted(set(_TOKEN.findall(text)))
        if tokens:
            raise ValueError(f"Unresolved template tokens in {policy.relative_path}: {tokens}")
    if policy.forbid_powershell_variables:
        variables = sorted(set(_POWERSHELL_VARIABLE.findall(text)))
        if variables:
            raise ValueError(f"Forbidden PowerShell variables in {policy.relative_path}: {variables}")
    return {
        "relative_path": policy.relative_path,
        "policy_class": policy.policy_class,
        "status": "PASS",
        "size_bytes": len(raw),
    }


def partC_rev1_validate_handoff_text_tree(
    root: str | Path,
    classification_manifest: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    base = Path(root).resolve(strict=True)
    policies = _load_classification_manifest(classification_manifest)
    results: list[dict[str, Any]] = []
    for policy in policies:
        candidate = (base / Path(*PurePosixPath(policy.relative_path).parts)).resolve(strict=False)
        try:
            candidate.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"Classified path escapes handoff root: {policy.relative_path}") from exc
        if policy.policy_class == "binary_or_external" and policy.disposition == "external_only":
            results.append(
                {
                    "relative_path": policy.relative_path,
                    "policy_class": policy.policy_class,
                    "status": "PASS_EXTERNAL_ONLY",
                    "size_bytes": policy.size_bytes,
                    "signature": policy.signature,
                }
            )
            continue
        if not candidate.is_file():
            raise ValueError(f"Classified handoff file is missing: {policy.relative_path}")
        if policy.policy_class == "binary_or_external":
            actual_size = candidate.stat().st_size
            if actual_size != policy.size_bytes:
                raise ValueError(f"Binary size mismatch for {policy.relative_path}")
            results.append(
                {
                    "relative_path": policy.relative_path,
                    "policy_class": policy.policy_class,
                    "status": "PASS_BINARY",
                    "size_bytes": actual_size,
                    "signature": policy.signature,
                }
            )
        else:
            results.append(_validate_text(candidate, policy))
    return results
