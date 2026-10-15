"""Checks that two versions of a Python file differ only in docstrings and comments.

    python verification/integrity_checks/ast_equivalence.py OLD.py NEW.py

The abstract syntax trees are compared after removing docstrings; comments are not
part of the AST. Identical trees mean that the executable code is unchanged."""
import ast, sys

def strip(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree, annotate_fields=True, include_attributes=False)

old, new = (strip(ast.parse(open(p).read())) for p in sys.argv[1:3])
print(("IDENTICAL CODE: " if old == new else "CODE DIFFERS:   ") + sys.argv[2])
sys.exit(0 if old == new else 1)
