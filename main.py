import yaml

with open("config/arima_config.yaml", "r") as f:
    config = yaml.safe_load(f)

order = tuple(config["model"]["order"])
