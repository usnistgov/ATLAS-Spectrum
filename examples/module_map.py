"""module_map.py: map every import and find code that exists twice.

Parses each .py file in the repo with `ast` (nothing is imported or executed),
then reports four things:

    1. every module, its imports, and what it defines
    2. which imports are shared, and which appear exactly once
    3. the internal dependency graph
    4. functions that exist in more than one module, split into
       identical bodies (should be shared) and same-name-different-body
       (should be renamed or reconciled)

    python examples/module_map.py            # report to stdout
    python examples/module_map.py --write    # also write docs/MODULE_MAP.md
    python examples/module_map.py --check    # exit 1 if identical duplicates exist

The --check form is the useful one in CI: it fails when the same function body
is copied into a second module.
"""

import argparse
import ast
import collections
import difflib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "docs", "MODULE_MAP.md")

SKIP = {".git", "__pycache__", ".venv", "venv", "node_modules", "downloads"}
STDLIB = set(sys.stdlib_module_names)

# Trivial bodies that are duplicated for good reason: a two-line CLI shim or a
# one-line accessor is not worth a shared import.
TRIVIAL_LINES = 4
# How alike two bodies must be before they count as the same thing written
# twice. 0.75 catches the real cases without flagging every pair of loops.
NEAR = 0.75
# Entry points and test scaffolding are meant to exist once per module.
EXPECTED_PER_MODULE = {"main", "check"}


def find_modules(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP]
        for n in sorted(filenames):
            if n.endswith(".py"):
                out.append(os.path.join(dirpath, n))
    return sorted(out)


def norm(node):
    """A function body as comparable text, with names kept but formatting,
    comments and the docstring dropped. ast.unparse gives us that for free.

    Size is real source lines: counting top-level statements would call an
    eight-line for-loop a one-line function.
    """
    body = list(node.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]                      # ignore the docstring
    span = (getattr(node, "end_lineno", node.lineno) or node.lineno) - node.lineno + 1
    return "\n".join(ast.unparse(b) for b in body), span


def parse(path):
    with open(path, "r", errors="replace") as f:
        src = f.read()
    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError as e:
        return {"error": f"{type(e).__name__}: {e}"}
    imports, funcs, classes = collections.defaultdict(set), {}, []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imports[a.name.split(".")[0]].add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:                   # relative import
                imports[("." * node.level) + (node.module or "")].add("(relative)")
            elif node.module:
                imports[node.module.split(".")[0]].add(
                    node.module + "." + ", ".join(a.name for a in node.names))
    for node in tree.body:                   # top level only
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            text, n = norm(node)
            funcs[node.name] = {"lines": n, "body": text,
                                "args": [a.arg for a in node.args.args],
                                "line": node.lineno}
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
    return {"imports": dict(imports), "funcs": funcs, "classes": classes,
            "loc": len(src.splitlines())}


def classify_import(name, local_names):
    if name.startswith("."):
        return "local"
    if name in local_names:
        return "local"
    if name in STDLIB:
        return "stdlib"
    return "third-party"


def analyse(root):
    paths = find_modules(root)
    mods = {}
    for p in paths:
        rel = os.path.relpath(p, root).replace(os.sep, "/")
        mods[rel] = parse(p)
    local_names = {os.path.splitext(os.path.basename(r))[0] for r in mods}

    usage = collections.defaultdict(set)     # import name -> modules using it
    for rel, m in mods.items():
        for name in m.get("imports", {}):
            usage[name].add(rel)

    # Every pair of non-trivial functions in different modules. Doing this
    # pairwise matters: a helper shared by two of three modules is invisible
    # to any check that asks whether all definitions of a name agree.
    everything = []
    for rel, m in mods.items():
        for fn, info in m.get("funcs", {}).items():
            if info["lines"] > TRIVIAL_LINES:
                everything.append((rel, fn, info))
    exact, near, seen_pair = [], [], set()
    for i, (rel_a, fn_a, a) in enumerate(everything):
        for rel_b, fn_b, b in everything[i + 1:]:
            if rel_a == rel_b:
                continue
            if fn_a in EXPECTED_PER_MODULE and fn_b in EXPECTED_PER_MODULE:
                continue
            ratio = difflib.SequenceMatcher(None, a["body"], b["body"]).ratio()
            if ratio < NEAR:
                continue
            key = tuple(sorted([(rel_a, fn_a), (rel_b, fn_b)]))
            if key in seen_pair:
                continue
            seen_pair.add(key)
            (exact if a["body"] == b["body"] else near).append(
                (ratio, rel_a, fn_a, a, rel_b, fn_b, b))
    exact.sort(key=lambda x: -x[3]["lines"])
    near.sort(key=lambda x: -x[0])
    return mods, usage, local_names, exact, near


def render(mods, usage, local_names, exact, near):
    L = []
    add = L.append
    total_loc = sum(m.get("loc", 0) for m in mods.values())
    add("# Module and import map\n")
    add(f"Generated by `python examples/module_map.py`. "
        f"{len(mods)} modules, {total_loc:,} lines.\n")

    add("\n## 1. Modules\n")
    add("| Module | Lines | Imports | Defines |")
    add("|---|---:|---|---|")
    for rel in sorted(mods):
        m = mods[rel]
        if "error" in m:
            add(f"| `{rel}` | ? | PARSE ERROR | {m['error']} |")
            continue
        imps = ", ".join(f"`{i}`" for i in sorted(m["imports"])) or "none"
        defines = len(m["funcs"]) + len(m["classes"])
        add(f"| `{rel}` | {m['loc']} | {imps} | "
            f"{defines} ({len(m['classes'])} class) |")

    add("\n## 2. Imports by reach\n")
    groups = collections.defaultdict(list)
    for name, users in usage.items():
        groups[classify_import(name, local_names)].append((name, users))
    for kind in ("stdlib", "third-party", "local"):
        rows = sorted(groups.get(kind, []), key=lambda x: (-len(x[1]), x[0]))
        if not rows:
            continue
        add(f"\n### {kind}\n")
        add("| Import | Used by | Modules |")
        add("|---|---:|---|")
        for name, users in rows:
            who = ", ".join(f"`{u}`" for u in sorted(users))
            add(f"| `{name}` | {len(users)} | {who} |")

    add("\n## 3. Internal dependency graph\n")
    add("```")
    edges = 0
    for rel in sorted(mods):
        m = mods[rel]
        deps = sorted(i for i in m.get("imports", {})
                      if classify_import(i, local_names) == "local")
        if deps:
            edges += len(deps)
            add(f"{rel:28} -> {', '.join(deps)}")
    if not edges:
        add("(no module imports another)")
    add("```")

    add("\n## 4. Duplicated definitions\n")
    if exact:
        add("### Identical bodies: one shared definition would do\n")
        add("| Lines | A | B |")
        add("|---:|---|---|")
        for _r, ra, fa, a, rb, fb, b in exact:
            add(f"| {a['lines']} | `{ra}`:{a['line']} `{fa}()` | "
                f"`{rb}`:{b['line']} `{fb}()` |")
    else:
        add("No function body is duplicated across modules.\n")
    if near:
        add("\n### Near-identical: the same job written twice\n")
        add("| Similarity | A | B |")
        add("|---:|---|---|")
        for ratio, ra, fa, a, rb, fb, b in near:
            add(f"| {ratio:.0%} | `{ra}`:{a['line']} `{fa}()` | "
                f"`{rb}`:{b['line']} `{fb}()` |")
    else:
        add("\nNo near-identical function bodies.\n")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true",
                    help=f"write the report to {os.path.relpath(OUT, ROOT)}")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any function body is duplicated")
    args = ap.parse_args()

    mods, usage, local_names, exact, near = analyse(ROOT)
    text = render(mods, usage, local_names, exact, near)
    if args.write:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as f:
            f.write(text)
        print(f"wrote {os.path.relpath(OUT, ROOT)}")
    else:
        print(text)
    if args.check:
        if exact:
            print(f"\n{len(exact)} pair(s) duplicated verbatim across modules:",
                  file=sys.stderr)
            for _r, ra, fa, _a, rb, fb, _b in exact:
                print(f"  {fa}(): {ra} and {rb}", file=sys.stderr)
            return 1
        if near:
            # Advisory: structural similarity is not always duplication. Two
            # readers of different CSV shapes look alike and should stay apart.
            print(f"note: {len(near)} near-identical pair(s), see the report")
        print("no function body is duplicated across modules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
