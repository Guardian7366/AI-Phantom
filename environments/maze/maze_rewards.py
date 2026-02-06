from typing import Any, Dict
import numpy as np

REWARD_SUCCESS = 1.0          #Recompensa terminal por alcanzar objetivo
REWARD_STEP = -0.01           #Penalización por paso (tiempo)
REWARD_COLLISION = -0.1       #Penalización por intentar moverse contra pared
REWARD_PROGRESS = 0.05        #Recompensa por reducción neta de distancia (shaping suave)
PENALTY_REGRESS = -0.02       #Penalización por aumentar distancia al objetivo
PENALTY_REVISIT = -0.05       #Penalización por visitar repetidamente la misma celda
REVISIT_THRESHOLD = 3         #Umbral (visitas) para empezar a penalizar

def compute_reward(
        state: Any,
        action: int,
        next_state: Any,
        info: Dict[str, Any],
        done: bool
) -> float :
    #Definir recompensa en 0
    reward = 0.0

    reward += REWARD_STEP #Penalización ligera por cada paso realizado

    #Obtener valor booleano que define si el agente ha colisionado con un muro al avanzar
    collided = bool(info.get("collided", False))
    if collided:
        #Penalización por colisión
        reward += REWARD_COLLISION
    
    #Distancia previa y actual del objetivo a encontrar
    prev_distance = info.get("prev_distance", None)
    cur_distance = info.get("distance_to_goal", None)

    #Define que ambas distancias deben existir para poder comparar y recompensar correctamente
    if prev_distance is not None and cur_distance is not None:
        if cur_distance < prev_distance:
            reward += REWARD_PROGRESS #Recompensa por tener una distancia más cercana al objetivo
        elif prev_distance < cur_distance:
            reward += PENALTY_REGRESS #Penalización por tener una distancia más alejada

    #Determina si el agente a terminado el episodio por encontrar el objetivo
    if done and info.get("terminal_type") == "success":
        reward = max(reward, 0.0) #Evita que el paso penalice más que success
        reward += REWARD_SUCCESS #Recompensa por alcanzar el objetivo

    #Recupera el número de visitas en la posición actual del fantasma
    visited_count = int(info.get("visited_count", 0))
    #Determina si la cantidad de visitas a la posición supera el umbral establecido
    if visited_count > REVISIT_THRESHOLD:
        reward += PENALTY_REVISIT * (visited_count - REVISIT_THRESHOLD) #Calcula penallización según cantidad de viisitas sobre el límite
    
    return float(reward)

