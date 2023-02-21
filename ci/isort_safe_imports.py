#!/usr/bin/env python
# Copyright (c) 2023, NVIDIA CORPORATION.

import ast
import re
import typing
from pathlib import Path

import click
import isort

RE_SAFE_IMPORT_CALL = r"(?:safe_import|(:?cpu|gpu)_only_import)(?P<from>_from)?"

RE_SAFE_IMPORT = r"\S+\s=\s*" + RE_SAFE_IMPORT_CALL


def find_root_imports(source: str):
    """Identify root import statements in the source code."""

    tree = ast.parse(source)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node


def is_safe_import(node):
    """Identify safe import assignments in the source code."""

    def func_name(call):
        try:
            return call.func.id
        except AttributeError:
            return call.func.attr

    return (
        isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and re.match(RE_SAFE_IMPORT_CALL, func_name(node.value))
    )


def find_root_safe_imports(source: str):
    """Find all root safe import assignments in source."""

    tree = ast.parse(source)
    for node in ast.iter_child_nodes(tree):
        if is_safe_import(node):
            yield node


def extract_safe_imports(source: str) -> typing.Tuple[str, list[str], int]:
    """Extract all safe import assignments and return them separately."""

    safe_import_nodes = list(find_root_safe_imports(source))

    # Extract source code for safe import assignments ...
    safe_imports = []
    for node in safe_import_nodes:
        safe_imports.append(ast.get_source_segment(source, node))

    # ... and remove them from the current source code.
    lines = source.split("\n")
    for node in reversed(safe_import_nodes):
        for i in reversed(range(node.lineno - 1, node.end_lineno)):
            lines.pop(i)

    # We identify the line number of the first safe import assignment which can
    # be used as a location to re-insert the block again later after
    # reformatting.
    lineno_first_safe_import = i if safe_import_nodes else -1

    return "\n".join(lines), safe_imports, lineno_first_safe_import


def _unwrap_safe_import(
    node: ast.Assign,
) -> typing.Union[ast.Import, ast.ImportFrom]:
    """Translate a safe import assignment into a standard import statement."""

    module = node.value.args[0].value

    assert len(node.targets) == 1
    target = node.targets[0]
    identifier = target.id

    if "from" in node.value.func.id:
        symbol = node.value.args[1].value
        if identifier == symbol:
            name = ast.alias(symbol)
        else:
            name = ast.alias(symbol, identifier)
        return ast.ImportFrom(
            module=module,
            names=[name],
            level=0,
        )
    else:
        if module == identifier:
            name = ast.alias(module)
        else:
            name = ast.alias(module, identifier)
        return ast.Import(names=[name])


class SafeImportUnwrapper(ast.NodeTransformer):
    def visit_Assign(self, node: ast.Assign) -> typing.Any:
        if is_safe_import(node):
            return _unwrap_safe_import(node)
        return node


def unwrap_safe_import(source: str) -> str:
    return ast.unparse(SafeImportUnwrapper().visit(ast.parse(source)))


def isort_safe_imports(safe_imports: list[str]) -> str:
    """Sort a list of safe import assignments with isort."""

    # Translate all safe import assignments to standard import statements.
    mapping = dict()
    for safe_import_stmt in safe_imports:
        mapping[unwrap_safe_import(safe_import_stmt)] = safe_import_stmt

    # Reformat the standard import statements with isort.
    standard_import_stmts = "\n".join(mapping.keys())
    isorted_standard_import_stmts = isort.code(standard_import_stmts, line_length=1e6)

    # Replace the isorted standard import statements with safe import assignments.
    reformatted_src = isorted_standard_import_stmts
    for std_import_stmt, safe_import_stmt in mapping.items():
        reformatted_src = reformatted_src.replace(std_import_stmt, safe_import_stmt)

    return reformatted_src


def format_safe_imports_block(source: str) -> typing.Optional[str]:
    """Format a block of safe import assignments with isort.

    For example, the following block:

        np = cpu_only_import("numpy")
        host_xpy = safe_import("numpy", alt=cp)
        cp = gpu_only_import("cupy")
        cudf = gpu_only_import("cudf")
        cached_property = safe_import_from(
            "functools", "cached_property", alt=null_decorator
        )

    will be reformatted to

        cached_property = safe_import_from(
            "functools", "cached_property", alt=null_decorator
        )

        cudf = gpu_only_import("cudf")
        cp = gpu_only_import("cupy")
        np = cpu_only_import("numpy")
    """

    source_wo_safe_imports, safe_imports, block_start = extract_safe_imports(source)
    if not safe_imports:
        return

    # Split source w/o safe imports on line-breaks while preserving the terminal
    # line.
    lines = source_wo_safe_imports.split("\n")

    # Override block_start with first line after root module imports if possible.
    root_imports = list(find_root_imports(source_wo_safe_imports))
    if root_imports:
        block_start = max(node.end_lineno for node in root_imports)

    # Remove all pre-existing whitespace after block start.
    while block_start < len(lines) and not lines[block_start].strip():
        lines.pop(block_start)

    # Sort the safe_imports block with isort.
    isorted_safe_imports = isort_safe_imports(safe_imports).splitlines()

    # Insert safe imports block into source code with two following empty lines.
    # Note that the insertion is done in reverse order to avoid the need to
    # shift the insertion index (block_start).
    lines.insert(block_start, "")
    lines.insert(block_start, "")
    for line in reversed(isorted_safe_imports):
        lines.insert(block_start, line)

    # Insert one empty line before block if needed.
    if block_start > 0 and lines[block_start - 1].strip():
        lines.insert(block_start, "")

    return "\n".join(lines)


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False), nargs=-1)
@click.option("-i", "--in-place", is_flag=True)
def main(input, in_place):
    for inp in input:
        path = Path(inp)
        source = path.read_text()

        try:
            reformatted_source = format_safe_imports_block(source)
        except SyntaxError:
            raise RuntimeError(f"Unable to parse {inp} .")
        except Exception as error:
            raise RuntimeError(f"Unable to reformat {inp} due to error: {error}")

        if reformatted_source and reformatted_source != source:
            if in_place:
                path.write_text(reformatted_source)
            else:
                click.echo(reformatted_source)


if __name__ == "__main__":
    main()
