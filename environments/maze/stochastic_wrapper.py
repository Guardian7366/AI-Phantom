import random


class StochasticMazeWrapper:
    """
    Wrapper que introduce ruido estocástico en las transiciones.

    Con probabilidad `action_noise_prob`,
    se reemplaza la acción del agente por una acción aleatoria.
    """

    def __init__(self, base_env, action_noise_prob: float = 0.1):
        self.base_env = base_env
        self.action_noise_prob = action_noise_prob

        # Exponer propiedades necesarias
        self.state_dim = base_env.state_dim
        self.action_space_n = base_env.action_space_n
        self.max_steps = base_env.max_steps

        # Factory compatible con arquitectura actual
        self.factory = lambda: StochasticMazeWrapper(
            self.base_env.factory(),
            action_noise_prob=self.action_noise_prob,
        )

    # -------------------------------------------------
    # Core API
    # -------------------------------------------------

    def reset(self):
        return self.base_env.reset()

    def step(self, action: int):
        if random.random() < self.action_noise_prob:
            action = random.randrange(self.action_space_n)

        return self.base_env.step(action)
