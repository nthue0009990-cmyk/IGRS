"""Pure inventory of explicit solution commands in named static DSS scripts.

This follows the pinned C-API reader's first-column block-comment rule. It is
not an interpreter for arbitrary DSS scripts: only literal Redirect/Compile
paths are supported, with no CD/DataPath changes, macros, or generated scripts.
No engine is imported or called and no engine event is synthesized.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re


class DSSCommandInventoryError(ValueError):
    """The named static script cannot be inventoried without assumptions."""


def _without_line_comment(line: str) -> str:
    result, quote, index = [], None, 0
    while index < len(line):
        character = line[index]
        if quote:
            result.append(character)
            if character == quote:
                if index + 1 < len(line) and line[index + 1] == quote:
                    result.append(line[index + 1])
                    index += 1
                else:
                    quote = None
        elif character in ("'", '"'):
            quote = character
            result.append(character)
        elif character == "!" or line[index:index + 2] == "//":
            break
        else:
            result.append(character)
        index += 1
    if quote:
        raise DSSCommandInventoryError("Unterminated quote in a static command line")
    return "".join(result).strip()


def _literal_include(payload: str) -> str:
    payload = re.sub(r"^file\s*=\s*", "", payload.strip(), flags=re.IGNORECASE)
    if not payload:
        raise DSSCommandInventoryError("Include command has no literal filename")
    pairs = {'"': '"', "'": "'", "[": "]"}
    if payload[0] in pairs:
        closing = pairs[payload[0]]
        end = payload.find(closing, 1)
        if end < 0 or payload[end + 1:].strip():
            raise DSSCommandInventoryError("Unsupported include filename or trailing options")
        value = payload[1:end]
    else:
        parts = payload.split()
        if len(parts) != 1:
            raise DSSCommandInventoryError("Include filenames containing spaces must be quoted")
        value = parts[0]
    if not value or any(marker in value for marker in ("@", "$", "%")):
        raise DSSCommandInventoryError("Dynamic include paths are outside the named static scope")
    return value


def inventory_dss_commands(entry_file: str | Path, *, allowed_root: str | Path | None = None) -> dict:
    """Return file hashes, include chains and separate Solve/CalcVoltageBases counts.

    The entry file is treated as a single Compile. Redirect temporarily uses the
    included file's directory and restores the caller directory; Compile leaves
    its directory for subsequent relative includes. Repeated includes count
    repeatedly. A recursive include cycle or unsupported path semantics rejects
    the inventory, rather than returning an apparently complete budget.
    """
    entry = Path(entry_file).resolve()
    root = Path(allowed_root).resolve() if allowed_root is not None else entry.parent
    files, commands, ignored, visits = {}, [], [], []
    counts = {name: 0 for name in ("solve", "calcvoltagebases", "redirect", "compile")}

    def visit(path, chain, active):
        path = path.resolve()
        if not path.is_relative_to(root):
            raise DSSCommandInventoryError("Include escapes the declared source root: " + str(path))
        if path in active:
            raise DSSCommandInventoryError("Recursive include cycle: " + " -> ".join(str(item) for item in (*active, path)))
        if not path.is_file():
            raise DSSCommandInventoryError("Included file is missing: " + str(path))
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DSSCommandInventoryError("Named static source is not UTF-8: " + str(path)) from exc
        identifier = path.relative_to(root).as_posix()
        if identifier in files and files[identifier]["sha256"] != digest:
            raise DSSCommandInventoryError("Source changed while building inventory: " + identifier)
        files[identifier] = {"path": str(path), "sha256": digest, "bytes": len(content)}
        visits.append({"file": identifier, "sha256": digest, "include_chain": list(chain)})
        in_block, block_start, current_directory = False, None, path.parent
        for line_number, line in enumerate(text.splitlines(), 1):
            if not in_block and line.startswith("/*"):
                in_block, block_start = True, line_number
            if in_block:
                candidate = line.strip().split(None, 1)
                if candidate and candidate[0].lower() in ("solve", "calcvoltagebases"):
                    ignored.append({"file": identifier, "line": line_number, "sha256": digest,
                                    "keyword": candidate[0].lower(), "reason": "FIRST_COLUMN_BLOCK_COMMENT",
                                    "block_start_line": block_start, "include_chain": list(chain)})
                if "*/" in line:
                    in_block = False
                continue
            code = _without_line_comment(line)
            if not code:
                continue
            parts = code.split(None, 1)
            keyword, payload = parts[0].lower(), parts[1] if len(parts) == 2 else ""
            if keyword == "cd" or (keyword == "set" and re.match(r"datapath\s*=", payload, re.IGNORECASE)):
                raise DSSCommandInventoryError("CD/DataPath changes are outside the named static scope")
            if keyword not in counts:
                continue
            record = {"file": identifier, "line": line_number, "sha256": digest,
                      "keyword": keyword, "text": code, "include_chain": list(chain)}
            counts[keyword] += 1
            commands.append(record)
            if keyword in ("redirect", "compile"):
                target = (current_directory / _literal_include(payload)).resolve()
                record["target"] = str(target)
                edge = {"file": identifier, "line": line_number, "sha256": digest,
                        "keyword": keyword, "target": str(target)}
                visit(target, (*chain, edge), (*active, path))
                record["target_sha256"] = files[target.relative_to(root).as_posix()]["sha256"]
                if keyword == "compile":
                    current_directory = target.parent
        # The engine ends a block comment at EOF; it does not leak into callers.
        if in_block:
            ignored.append({"file": identifier, "line": block_start, "sha256": digest,
                            "reason": "BLOCK_COMMENT_ENDS_AT_FILE_EOF", "include_chain": list(chain)})

    visit(entry, (), ())
    return {"status": "STATIC_NAMED_DSS_COMMAND_INVENTORY", "entry_file": str(entry), "entry_mode": "Compile",
            "allowed_root": str(root), "counts": counts, "explicit_solve_commands": counts["solve"],
            "calcvoltagebases_entries": counts["calcvoltagebases"],
            "command_triggered_numerical_solution_entries": counts["solve"] + counts["calcvoltagebases"],
            "files": files, "file_visits": visits, "commands": commands, "ignored_solution_tokens": ignored,
            "new_powerflow_calls": 0,
            "limitations": ["Literal exact-keyword static includes only; not an arbitrary DSS interpreter.",
                "Counts are command-triggered entries, not internal control loops, initialization solves, iterations or linear-system solves.",
                "A top-level Compile caller is represented by entry_mode, not added as a second script command.",
                "No numerical solution, feasibility result, or engine event is inferred or generated."]}
