from typing import Any, Dict, Tuple


class EnvironmentBase:
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        raise NotImplementedError

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        raise NotImplementedError
