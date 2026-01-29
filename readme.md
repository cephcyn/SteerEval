# SteerEval

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

### Overall pipeline description

1. Data setup: combines MovieLens 25M ratings with TMDb descriptions by default, and loads tag metadata. Drop items missing TMDb descriptions.
2. Profile creation: samples users with some conditions (we set threshold to >= 100 total reviews) and build profiles based on the first N (we use 50) items
3. Candidate set creation: based on a given steering task (increase or decrease) and steered tag, select 100 items (50 relevant to tag, 50 irrelevant to tag), with exactly 1 true next item (the item that the user actually watched next) and the remaining 99 being selected based on most similar TMDb popularity score to the true item
4. Pre-steering ("original") profile: construct the original (baseline) profile
5. Pre-steering ("original") ranking: rank the 100 candidate items based on original profile
6. Steering action: edit the original profile by applying increase/decrease tag templates or using LLM prompts
7. Post-steering ("steered") profile: we now have an updated (steered) profile
8. Post-steering ("steered") ranking: rank the 100 candidate items based on updated profile
9. Metrics: measure how tag-relevant items move between the pre-steering and post-steering rankings. Average these metrics across users, tags, and steering tasks as appropriate.

On our servers, running 10 steering scenarios took approximately 40 minutes on one A6000 GPU. One combination of user history, candidate set, steering intervention, and recommendation pipeline is one scenario. Each reported "increase" and "decrease" average delta-AUC metric for a given pipeline configuration was generally averaged over (94 tags x 10 users =) 940 steering scenarios.

### Need to add a steering intervention or recommendation component?

To use a different rating history or tag information dataset: refer to the functions defined in `src/data_load/` to see examples of how current datasets are loaded, and then refer to how they are called within `steering_eval.py` to see examples of how they are used. Feel free to add arguments to or adapt `steering_eval.py` to support new dataset loading as appropriate.

To use a new profile creation dataset: same as above, except functions are defined in `src/profile_create/`.

To use a new ranking method: same as above, except functions are defined in `src/recommend/`.

To use a new steering intervention: same as above, except functions are defined in `src/profile_edit/`.

To implement a new evaluation metric: same as above, except functions are defined in `src/score/`.

## Results and datasets

See [https://huggingface.co/collections/cephcyn/steereval-datasets-and-results](https://huggingface.co/collections/cephcyn/steereval-datasets-and-results) for dataset and experiment result files

## Citation

(Currently anonymous)
