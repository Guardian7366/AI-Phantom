import yaml


def load_yaml_config(path: str) -> dict:
    """
    Carga un archivo YAML y retorna el diccionario de configuración.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
