"""
Architecture test: verify dataset loader method contracts.

- load_split() must exist (it's the correct method)
- load() must NOT exist (it's the C4 bug)

This prevents regression of C4 (loader.load() used instead of load_split()).
"""

import importlib

EXPECTED_HAS = ["load_split", "load_raw_dataset", "convert_to_examples"]
MUST_NOT_HAVE = ["load"]

LOADER_MODULES = [
    ("src.core.loaders.conll.loader", "CONLLLoader"),
    ("src.core.loaders.biomedical.genia", "GENIALoader"),
]


def test_loaders_have_load_split():
    for module_path, class_name in LOADER_MODULES:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        for method in EXPECTED_HAS:
            assert hasattr(cls, method), (
                f"{class_name} missing method '{method}'"
            )


def test_loaders_do_not_have_load():
    for module_path, class_name in LOADER_MODULES:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        for method in MUST_NOT_HAVE:
            assert not hasattr(cls, method), (
                f"{class_name} has dangerous method '{method}' — use load_split() instead"
            )
