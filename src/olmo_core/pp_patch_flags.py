import os


FALSE_VALUES = {"0", "false", "no", "off"}
TRUE_VALUES = {"1", "true", "yes", "on"}


def pp_fixes_enabled() -> bool:
    """Whether local TP+PP compatibility patches should be active."""
    disable_value = os.environ.get("OLMO_CORE_DISABLE_PP_FIXES")
    if disable_value is not None:
        return disable_value.strip().lower() not in TRUE_VALUES

    enable_value = os.environ.get("OLMO_CORE_ENABLE_PP_FIXES")
    if enable_value is not None:
        return enable_value.strip().lower() not in FALSE_VALUES

    return True
