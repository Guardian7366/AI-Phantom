class AgentBase:
    def act(self, obs, deterministic: bool = False) -> int:
        raise NotImplementedError

    def learn(self) -> dict:
        raise NotImplementedError
