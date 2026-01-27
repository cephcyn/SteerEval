# steerabilityeval

## Setup

1. Use the Conda environment (or install appropriate pip packages directly) in `environment.yaml`
2. In the new environment, run additional package install commands:
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
3. Set up environment variables in `env.sh` as appropriate to access model weights

## Usage

Call:
```
python src/steering_eval.py
```
with appropriate arguments. (Arguments are documented within `steering_eval.py` itself.)

### Need to add a steering intervention or recommendation component?

To use a different rating history or tag information dataset: refer to the functions defined in `src/data_load/` to see examples of how current datasets are loaded, and then refer to how they are called within `steering_eval.py` to see examples of how they are used. Feel free to add arguments to or adapt `steering_eval.py` to support new dataset loading as appropriate.

To use a new profile creation dataset: same as above, except functions are defined in `src/profile_create/`.

To use a new ranking method: same as above, except functions are defined in `src/recommend/`.

To use a new steering intervention: same as above, except functions are defined in `src/profile_edit/`.

To implement a new evaluation metric: same as above, except functions are defined in `src/score/`.

## Citation

(Currently anonymous)
