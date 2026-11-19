from importlib import import_module


__all__ = [
    "BasePolicy",
    "Gr00tPolicy",
    "PolicyWrapper",
]


def __getattr__(name):
    if name == "Gr00tPolicy":
        module_name = ".gr00t_policy"
    elif name in {"BasePolicy", "PolicyWrapper"}:
        module_name = ".policy"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
