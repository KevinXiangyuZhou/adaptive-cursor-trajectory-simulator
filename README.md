# adaptive-cursor-trajectory-simulator

## Setup

Requires Python 3.8+.

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e hcs_package
pip install -r requirements.txt
```

## Run simulation experiment in pygame

```bash
python -m experiment.pygame_experiment \
  --env-config experiment/environment_configurations/sigmoidal_tunnel.json \
  --user-config hcs_package/src/hcs_package/user_configurations/office_worker.json
```
