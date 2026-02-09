from pathlib import Path
from typing import Callable, Optional

from .policy_common import run_policy_from_yaml

DEFAULT_YAML = Path("Panbot/config/policies.yaml")


def run_policy1(
    *,
    yaml_path: str | Path = DEFAULT_YAML,
    duration_s_override: float | None = None,
    stop_condition: Optional[Callable[[], bool]] = None,
) -> None:
    run_policy_from_yaml(
        yaml_path=yaml_path,
        policy_name="policy1",
        duration_s_override=duration_s_override,
        stop_condition=stop_condition,
    )
