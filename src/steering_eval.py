#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
import pickle
import os
import sys
import socket
import platform
import subprocess
import traceback
from pathlib import Path
import json
from datetime import datetime
import logging
import random
import pprint
from typing import Any, Mapping, Tuple

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM  # type: ignore
from sentence_transformers import SentenceTransformer

from custom_types import AddtNotesDict, template_addt_notes_dict
from custom_types import UserID, ItemID, TagID
from custom_types import template_item_info_dict, template_rating_data_df
from custom_types import template_tag_info_df, template_tag_links_df
import data_load.synthetic as data_synthetic
import data_load.movielens as data_movielens
import data_load.navigatingsensitivity as data_navigatingsensitivity
import data_load.tmdb as data_tmdb
from custom_types import SampleInfo
from misc import llm_tools, sample_setup
from custom_types import TextProfile
from misc.item_template import get_tag_names
import profile_create.naive as profile_create_naive
import profile_create.llm as profile_create_llm
import profile_edit.naive as profile_edit_naive
import profile_edit.llm as profile_edit_llm
from custom_types import RankingDF
import recommend.naive as recommend_naive
import recommend.embed as recommend_embed
import recommend.llm as recommend_llm
from score import metrics

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# -----------------------------------------------------------------------------
# Script arguments
# -----------------------------------------------------------------------------


def arg_bool(s: str):
    return s == "True"


def arg_int_list(s: str):
    return [int(e) for e in s.split(",") if (len(e.strip()) > 0)]


parser = argparse.ArgumentParser(
    description="Evaluate steering intervention for a given user sample size, steering task, tag, and pipeline implementation.",
)
# job name and filesystem
parser.add_argument(
    "--dataset_dir",
    type=str,
    default=os.environ.get(
        "STEEREVAL_WORKING_DIR",
        "../datasets/",
    ),
    help="directory to pull datasets from",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=os.environ.get(
        "STEEREVAL_OUTPUT_DIR",
        "../output/",
    ),
    help="parent output directory",
)
parser.add_argument(
    "--job_name",
    type=str,
    required=True,
    help="hyperspecific job name. used for writing cache and final result files within output parent dir.",
)
# caching and interaction
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="randomizer seed",
)
# parser.add_argument(
#     "--model_device",
#     type=str,
#     default=os.environ.get(
#         "STEEREVAL_MODEL_DEVICE",
#         "cuda",
#     ),
#     help="embed device to use for large models, if relevant",
# )
parser.add_argument(
    "--ask_manual_pauses",
    type=arg_bool,
    default=False,
    help="True/False: should the job pause to confirm at end of each stage?",
)
parser.add_argument(
    "--use_cache_dataset",
    type=arg_bool,
    default=False,
    help="True/False: use any existing cache files for dataset load?",
)
parser.add_argument(
    "--cache_dataset_dir",
    type=str,
    default="../output_partials/0_dataset/",
    help="directory to use for dataset load caches",
)
parser.add_argument(
    "--use_cache_sampling",
    type=arg_bool,
    default=False,
    help="True/False: use any existing cache files for user and item sampling?",
)
parser.add_argument(
    "--cache_sampling_dir",
    type=str,
    default="../output_partials/1_sampling/",
    help="directory to use for sampling caches",
)
parser.add_argument(
    "--use_cache_profile",
    type=arg_bool,
    default=False,
    help="True/False: use any existing cache files for original profile creation?",
)
parser.add_argument(
    "--cache_profile_dir",
    type=str,
    default="../output_partials/2_profiles/",
    help="directory to use for profile generation caches",
)
parser.add_argument(
    "--use_cache_profile_partial",
    type=arg_bool,
    default=False,
    help="True/False: use any existing cache files for original profile creation (partial progress)?",
)
parser.add_argument(
    "--use_cache_profile_partial_incr",
    type=int,
    default=10,
    help="interval to use between saving partial progress for original profile creation",
)
parser.add_argument(
    "--cache_profile_partial_dir",
    type=str,
    default="../output_partials/2_profiles_partial/",
    help="directory to use for partial profile generation caches",
)
parser.add_argument(
    "--use_cache_steering",
    type=arg_bool,
    default=False,
    help="True/False: use any existing cache files for steering intervention?",
)
parser.add_argument(
    "--cache_steering_dir",
    type=str,
    default="../output_partials/3_steering/",
    help="directory to use for steering intervention caches",
)
parser.add_argument(
    "--use_cache_steering_partial",
    type=arg_bool,
    default=False,
    help="True/False: use any existing cache files for steering intervention (partial progress)?",
)
parser.add_argument(
    "--use_cache_steering_partial_incr",
    type=int,
    default=10,
    help="interval to use between saving partial progress for steering intervention",
)
parser.add_argument(
    "--cache_steering_partial_dir",
    type=str,
    default="../output_partials/3_steering_partial/",
    help="directory to use for partial steering intervention caches",
)
parser.add_argument(
    "--use_cache_ranking",
    type=arg_bool,
    default=False,
    help="True/False: use any existing cache files for pre- and post-steering ranking?",
)
parser.add_argument(
    "--cache_ranking_dir",
    type=str,
    default="../output_partials/4_ranking/",
    help="directory to use for ranking caches",
)
parser.add_argument(
    "--use_cache_ranking_partial",
    type=arg_bool,
    default=False,
    help="True/False: use any existing cache files for pre- and post-steering ranking (partial progress)?",
)
parser.add_argument(
    "--use_cache_ranking_partial_incr",
    type=int,
    default=10,
    help="interval to use between saving partial progress for pre- and post-steering ranking",
)
parser.add_argument(
    "--cache_ranking_partial_dir",
    type=str,
    default="../output_partials/4_ranking_partial/",
    help="directory to use for partial ranking caches",
)
# dataset (sources)
parser.add_argument(
    "--data_user",
    type=str,
    choices=[
        # "synth", # TODO not integrated
        "movielens_25m",
    ],
    required=True,
    help="data source for user watch history",
)
parser.add_argument(
    "--data_tag",
    type=str,
    choices=[
        # "synth", # TODO not integrated
        "random",
        "movielens_25m_genre",
        "movielens_25m_genome",
        "doesthedogdie",
        "doesthedogdie_renamed",
    ],
    required=True,
    help="data source for item interaction tags",
)
# sampling (users, tags, and/or items)
parser.add_argument(
    "--sampling_target_user_ids",
    type=arg_int_list,
    default=arg_int_list(""),
    help=(
        "user ID(s) to use in steering task. "
        'empty means "pick random / use arguments". '
        "Overrides num_users, num_users_skipn arguments. "
        "Still subject to all other user sampling filters."
    ),
)
parser.add_argument(
    "--sampling_num_users",
    type=int,
    default=20,
    help="number of users to include in experiment",
)
parser.add_argument(
    "--sampling_num_users_skipn",
    type=int,
    default=0,
    help="skip the first N users when sampling",
)
parser.add_argument(
    "--sampling_user_min_ratings",
    type=int,
    default=0,
    help="number of minimum ratings for user to be eligible for task in eval",
)
parser.add_argument(
    "--sampling_history_size",
    type=int,
    default=30,
    help="number of items to use from user rating history (users with insufficient history are excluded)",
)
parser.add_argument(
    "--sampling_pool_num_related",
    type=int,
    default=10,
    help='number of related items to include in ranking task. -1 means "use all"',
)
parser.add_argument(
    "--sampling_pool_num_unrelated",
    type=int,
    default=10,
    help='number of unrelated items to include in ranking task. -1 means "use all"',
)
parser.add_argument(
    "--sampling_pool_num_next",
    type=int,
    default=1,
    help="number of user's upcoming items to include in ranking task. Must be nonnegative. May overlap with related or unrelated.",
)
parser.add_argument(
    "--sampling_steer_action",
    type=str,
    choices=[
        "increase",
        "decrease",
    ],
    required=True,
    help="Steering action",
)
parser.add_argument(
    "--sampling_steer_target_tag_ids",
    type=arg_int_list,
    default=arg_int_list(""),
    help='tag ID(s) to target in steering task. empty means "pick random"',
)
# profile (creation)
parser.add_argument(
    "--profile_method",
    type=str,
    choices=[
        "naive_empty",
        "naive_taglist",  # a string containing all relevant tags
        "list_movies",  # the raw movie history metadata sequence
        "llm_movies_sentence",  # llm generates sentence, based on movie sequence
        "llm_movies_paragraph",  # llm generates paragraph, based on movie sequence
    ],
    required=True,
    help="type of profile creation used",
)
parser.add_argument(
    "--profile_data",
    type=str,
    choices=[
        "all_tags",
        "title",
        "description",
        "title+description",
    ],
    required=True,
    help="data used for profile creation",
)
parser.add_argument(
    "--profile_llm_model",
    type=str,
    default=os.environ.get(
        "STEEREVAL_PROFILE_LLM_MODEL",
        "meta-llama/Llama-3.1-8B-Instruct",
        # "meta-llama/Llama-3.2-3B-Instruct",
    ),
    help="LLM model to use for profile creation, if relevant",
)
# steering (intervention)
parser.add_argument(
    "--steering_method",
    type=str,
    choices=[
        "naive_taglist",  # append something to string containing all relevant tags
        "append_template",
        "append_template_alt1",
        "append_template_alt2",
        "append_template_alt3_genre",
        "append_template_alt3_ddd",
        "append_template_alt4",
        "append_template_alt5_genre",
        "append_template_alt5_ddd",
        "append_template_althybrid",
        "append_template_alt_emph_weak_genre",
        "append_template_alt_emph_strong_genre",
        "append_template_alt_emph_weak_ddd",
        "append_template_alt_emph_strong_ddd",
        "append_llm",
        "append_llm_alt5_genre",
        "append_llm_alt5_ddd",
        "append_llm_althybrid",
        "rewrite_llm",
        "rewrite_llm_alt5_genre",
        "rewrite_llm_alt5_ddd",
        "rewrite_llm_althybrid",
        "replace",  # TODO naive template vs llm?
    ],
    required=True,
    help="type of steering intervention used",
)
parser.add_argument(
    "--steering_llm_model",
    type=str,
    default=os.environ.get(
        "STEEREVAL_STEERING_LLM_MODEL",
        "meta-llama/Llama-3.1-8B-Instruct",
        # "meta-llama/Llama-3.2-3B-Instruct",
    ),
    help="LLM model to use for steering intervention, if relevant",
)
# ranking (recommendation)
parser.add_argument(
    "--ranking_method",
    type=str,
    choices=[
        "naive_random",
        "naive_tagcount",
        "naive_oracle",
        "llm_ordering",
        "llm_scorepred",
        "embed_similarity",
    ],
    required=True,
    help="type of ranking used",
)
parser.add_argument(
    "--ranking_data",
    type=str,
    choices=[
        "all_tags",
        "target_tags",
        "title",
        "description",
        "title+description",
        "title+description+target_tags",
    ],
    required=True,
    help="data used for ranking",
)
parser.add_argument(
    "--ranking_llm_model",
    type=str,
    default=os.environ.get(
        "STEEREVAL_RANKING_LLM_MODEL",
        "meta-llama/Llama-3.1-8B-Instruct",
        # "meta-llama/Llama-3.2-3B-Instruct",
    ),
    help="LLM model to use for ranking, if relevant",
)
parser.add_argument(
    "--ranking_embed_model",
    type=str,
    default=os.environ.get(
        "STEEREVAL_RANKING_EMBED_MODEL",
        "mixedbread-ai/mxbai-embed-large-v1",
    ),
    help="LLM model to use for ranking, if relevant",
)
# scoring (evaluation)
parser.add_argument(
    "--eval_k",
    type=arg_int_list,
    default=arg_int_list("5,10,20"),
    help="by default, ranking eval cutoff. format: 5,10,15",
)
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# HuggingFace auth token for model loading
# not necessarily used for all evaluation pipelines
HF_TOKEN = os.environ.get("HF_TOKEN", "")


@dataclass
class ExtraArgs:
    partial_dir_dataset: str
    partial_dir_sampling: str
    partial_dir_profile: str
    partial_dir_profile_partial: str
    partial_dir_steering: str
    partial_dir_steering_partial: str
    partial_dir_ranking: str
    partial_dir_ranking_partial: str
    result_dir: str
    load_tmdb: bool = False
    use_llm_profile: bool = False
    use_llm_steering: bool = False
    use_llm_ranking: bool = False
    use_embed_ranking: bool = False


extra_args = ExtraArgs(
    partial_dir_dataset=os.environ.get(
        "STEEREVAL_PARTIAL_DIR_DATASET",
        args.cache_dataset_dir,
    ),
    partial_dir_sampling=os.environ.get(
        "STEEREVAL_PARTIAL_DIR_SAMPLING",
        args.cache_sampling_dir,
    ),
    partial_dir_profile=os.environ.get(
        "STEEREVAL_PARTIAL_DIR_PROFILE",
        args.cache_profile_dir,
    ),
    partial_dir_profile_partial=os.environ.get(
        "STEEREVAL_PARTIAL_DIR_PROFILE_PARTIAL",
        args.cache_profile_partial_dir,
    ),
    partial_dir_steering=os.environ.get(
        "STEEREVAL_PARTIAL_DIR_STEERING",
        args.cache_steering_dir,
    ),
    partial_dir_steering_partial=os.environ.get(
        "STEEREVAL_PARTIAL_DIR_STEERING_PARTIAL",
        args.cache_steering_partial_dir,
    ),
    partial_dir_ranking=os.environ.get(
        "STEEREVAL_PARTIAL_DIR_RANKING",
        args.cache_ranking_dir,
    ),
    partial_dir_ranking_partial=os.environ.get(
        "STEEREVAL_PARTIAL_DIR_RANKING_PARTIAL",
        args.cache_ranking_partial_dir,
    ),
    result_dir=os.environ.get(
        "STEEREVAL_RESULT_DIR",
        (args.output_dir + args.job_name + "/"),
    ),
)

# check argument agreement and set extra args...
# data?
match args.data_user:
    case "movielens_25m":
        extra_args.load_tmdb = True
# sampling?
# profile?
match args.profile_method:
    case "naive_empty":
        assert args.steering_method != "naive_taglist"
        assert args.ranking_method != "naive_tagcount"
    case "naive_taglist":
        assert args.profile_data == "all_tags"
    case "list_movies":
        assert args.ranking_method != "naive_tagcount"
    case "llm_movies_sentence":
        extra_args.use_llm_profile = True
        assert args.ranking_method != "naive_tagcount"
    case "llm_movies_paragraph":
        extra_args.use_llm_profile = True
        assert args.ranking_method != "naive_tagcount"
match args.profile_data:
    case "all_tags":
        pass
    case "title":
        assert args.profile_method != "naive_taglist"
    case "description":
        extra_args.load_tmdb = True
        assert args.profile_method != "naive_taglist"
    case "title+description":
        extra_args.load_tmdb = True
        assert args.profile_method != "naive_taglist"
    case "title+description+target_tags":
        extra_args.load_tmdb = True
        assert args.profile_method != "naive_taglist"
        assert args.ranking_method != "llm_ordering"
# steering?
match args.steering_method:
    case "naive_taglist":
        assert args.profile_method == "naive_taglist"
        assert args.ranking_method == "naive_tagcount"
    case "append_template":
        assert args.ranking_method != "naive_tagcount"
    case "append_template_alt1":
        assert args.ranking_method != "naive_tagcount"
    case "append_template_alt2":
        assert args.ranking_method != "naive_tagcount"
    case "append_template_alt3_genre":
        assert args.ranking_method != "naive_tagcount"
    case "append_template_alt3_ddd":
        assert args.ranking_method != "naive_tagcount"
    case "append_template_alt4":
        assert args.ranking_method != "naive_tagcount"
    case "append_template_alt5_genre":
        assert args.ranking_method != "naive_tagcount"
    case "append_template_alt5_ddd":
        assert args.ranking_method != "naive_tagcount"
    case "append_template_althybrid":
        assert args.ranking_method != "naive_tagcount"
    case "append_template_alt_emph_weak_genre":
        assert args.ranking_method != "naive_tagcount"
    case "append_template_alt_emph_weak_ddd":
        assert args.ranking_method != "naive_tagcount"
    case "append_template_alt_emph_strong_ddd":
        assert args.ranking_method != "naive_tagcount"
    case "append_template_alt_emph_strong_genre":
        assert args.ranking_method != "naive_tagcount"
    case "append_llm":
        extra_args.use_llm_steering = True
        assert args.ranking_method != "naive_tagcount"
    case "append_llm_alt5_genre":
        extra_args.use_llm_steering = True
        assert args.ranking_method != "naive_tagcount"
    case "append_llm_alt5_ddd":
        extra_args.use_llm_steering = True
        assert args.ranking_method != "naive_tagcount"
    case "append_llm_althybrid":
        extra_args.use_llm_steering = True
        assert args.ranking_method != "naive_tagcount"
    case "rewrite_llm":
        extra_args.use_llm_steering = True
        assert args.ranking_method != "naive_tagcount"
    case "rewrite_llm_alt5_genre":
        extra_args.use_llm_steering = True
        assert args.ranking_method != "naive_tagcount"
    case "rewrite_llm_alt5_ddd":
        extra_args.use_llm_steering = True
        assert args.ranking_method != "naive_tagcount"
    case "rewrite_llm_althybrid":
        extra_args.use_llm_steering = True
        assert args.ranking_method != "naive_tagcount"
    case "replace":
        assert args.ranking_method != "naive_tagcount"
# ranking?
match args.ranking_method:
    case "naive_random":
        pass
    case "naive_tagcount":
        assert args.profile_method == "naive_taglist"
        assert args.steering_method == "naive_taglist"
        assert args.ranking_data == "all_tags"
    case "naive_oracle":
        pass
    case "llm_ordering":
        extra_args.use_llm_ranking = True
        assert args.profile_data != "title+description+target_tags"
    case "llm_scorepred":
        extra_args.use_llm_ranking = True
    case "embed_similarity":
        extra_args.use_embed_ranking = True
match args.ranking_data:
    case "all_tags":
        pass
    case "target_tags":
        assert args.ranking_method != "naive_tagcount"
    case "title":
        assert args.ranking_method != "naive_tagcount"
    case "description":
        extra_args.load_tmdb = True
        assert args.ranking_method != "naive_tagcount"
    case "title+description":
        extra_args.load_tmdb = True
        assert args.ranking_method != "naive_tagcount"
# general cache stuff
if args.use_cache_dataset:
    assert args.cache_dataset_dir is not None
if args.use_cache_sampling:
    assert args.cache_sampling_dir is not None
if args.use_cache_profile:
    assert args.cache_profile_dir is not None
if args.use_cache_profile_partial:
    assert args.cache_profile_partial_dir is not None
if args.use_cache_steering:
    assert args.cache_steering_dir is not None
if args.use_cache_steering_partial:
    assert args.cache_steering_partial_dir is not None
if args.use_cache_ranking:
    assert args.cache_ranking_dir is not None
if args.use_cache_ranking_partial:
    assert args.cache_ranking_partial_dir is not None

# reproducibility
random.seed(args.seed + 1)
rng_used = random.Random(args.seed)
np.random.seed(args.seed)
try:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
except Exception:
    pass

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------

Path(extra_args.partial_dir_dataset).mkdir(parents=True, exist_ok=True)
Path(extra_args.partial_dir_sampling).mkdir(parents=True, exist_ok=True)
Path(extra_args.partial_dir_profile).mkdir(parents=True, exist_ok=True)
Path(extra_args.partial_dir_profile_partial).mkdir(parents=True, exist_ok=True)
Path(extra_args.partial_dir_steering).mkdir(parents=True, exist_ok=True)
Path(extra_args.partial_dir_steering_partial).mkdir(parents=True, exist_ok=True)
Path(extra_args.partial_dir_ranking).mkdir(parents=True, exist_ok=True)
Path(extra_args.partial_dir_ranking_partial).mkdir(parents=True, exist_ok=True)
Path(extra_args.result_dir).mkdir(parents=True, exist_ok=True)

fname_logging = os.path.join(extra_args.result_dir, "log.txt")
Path(fname_logging).touch(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%m-%d %H:%M:%S",
    filename=fname_logging,
    filemode="w",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s"))
logging.getLogger("").addHandler(console)


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    logging.info(f"[{ts}] {msg}")
    # would like to verify that python logging reliably flushes memory in slurm
    # before removing the console print statements
    # print(f"[{ts}] {msg}", flush=True)


# Build our CSV to contain all results
# maps from row (user) identifier, to col name, to value
csv_export_peruser_dict: dict[Any, dict[str, Any]] = {}

# -----------------------------------------------------------------------------
# Script initial logging starts here
# -----------------------------------------------------------------------------

log("BEGINNING SCRIPT")

metadata = {
    "host": socket.gethostname(),
    "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
    "python": platform.python_version(),
    "cwd": os.getcwd(),
    "script": os.path.abspath(sys.argv[0]),
    "args": [vars(args), vars(extra_args)],
    "env": {
        k: os.environ.get(k, "")
        for k in ["HF_HOME", "TRANSFORMERS_CACHE", "CUDA_VISIBLE_DEVICES", "PYTHONPATH"]
    },
    "git_hash": "(no git)",
    "start_ts": datetime.now().isoformat(timespec="seconds"),
    "end_ts": "",
    "exit_code": 0,
}
try:
    metadata["git_hash"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()
except Exception:
    pass


def dump_metadata():
    log(json.dumps(metadata, indent=2))
    fname_metadata = os.path.join(extra_args.result_dir, "run_meta.json")
    Path(fname_metadata).touch(exist_ok=True)
    with open(fname_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


dump_metadata()


def maybe_pause(label: str):
    if args.ask_manual_pauses and sys.stdin.isatty():
        input(f"[pause] {label} - press Enter...")


# set up variables to hold models in memory
preloaded_model_autolm_tokenizer = None
preloaded_model_autolm = None
preloaded_model_senttf = None

# -----------------------------------------------------------------------------
# Script run: dataset creation
# -----------------------------------------------------------------------------

log(f"PHASE 0: datasets")

fname_cache = Path(extra_args.partial_dir_dataset) / "00_dataset.pkl"
fname_cache_offset = Path(extra_args.partial_dir_dataset) / "00_dataset.tmp.pkl"

df_ratings = template_rating_data_df()
item_info = template_item_info_dict()
removed_list: list[ItemID] = []
df_taginfo = template_tag_info_df()
df_taglinks = template_tag_links_df()
# check load first
need_to_generate = True
try:
    if args.use_cache_dataset and os.path.exists(fname_cache):
        log(f"loading complete results from existing: {fname_cache}")
        with open(fname_cache, "rb") as f:
            (
                df_ratings,
                item_info,
                removed_list,
                df_taginfo,
                df_taglinks,
                args_seed,
                args_dataset_dir,
                args_data_user,
                eargs_load_tmdb,
                args_data_tag,
            ) = pickle.load(f)
        # check arg compatibility
        need_to_generate = False
        if args.seed != args_seed:
            need_to_generate = True
        elif args.dataset_dir != args_dataset_dir:
            need_to_generate = True
        elif args.data_user != args_data_user:
            need_to_generate = True
        elif extra_args.load_tmdb != eargs_load_tmdb:
            need_to_generate = True
        elif args.data_tag != args_data_tag:
            need_to_generate = True
except Exception as e:
    log("".join(traceback.format_exception(e)))

try:
    if not need_to_generate:
        log(f"loading worked, args agree, not generating anew")
    else:
        log(f"constructing at least partially anew")
        match args.data_user:
            case "movielens_25m":
                df_ratings, item_info = data_movielens.movielens_25m_ratings_items(
                    raw_dir_path=args.dataset_dir + "movielens_25m",
                    allow_mkdir_downloads=False,
                )
                # add tmdb data if needed
                if extra_args.load_tmdb:
                    item_info, removed_list = data_tmdb.merge_tmdb_into_movielens(
                        tmdb_pkl_fname=args.dataset_dir + "tmdb/tmdb_data.pkl",
                        original_item_info=item_info,
                    )
            case _:
                raise NotImplementedError
        match args.data_tag:
            case "random":
                df_taginfo, df_taglinks = data_synthetic.random_tag_genres(
                    item_info=item_info,
                    random_rate=0.5,
                    rng=rng_used,
                    seed=args.seed,  # NOTE this is overridden by above arg
                )
            case "movielens_25m_genre":
                df_taginfo, df_taglinks = data_movielens.movielens_tag_genres(
                    raw_dir_path=args.dataset_dir + "movielens_25m",
                )
            case "movielens_25m_genome":
                df_taginfo, df_taglinks = data_movielens.movielens_tag_genome(
                    raw_dir_path=args.dataset_dir + "movielens_25m",
                )
            case "doesthedogdie":
                df_taginfo, df_taglinks = (
                    data_navigatingsensitivity.navigatingsensitivity_tags_ddd(
                        raw_dir_path=args.dataset_dir + "navigatingsensitivity",
                        out_dir_path=args.output_dir + "tag_ddd",
                        allow_mkdir_downloads=False,
                        get_movielens_analysis=False,
                    )
                )
            case "doesthedogdie_renamed":
                df_taginfo, df_taglinks = (
                    data_navigatingsensitivity.navigatingsensitivity_tags_ddd(
                        raw_dir_path=args.dataset_dir + "navigatingsensitivity",
                        out_dir_path=args.output_dir + "tag_ddd",
                        allow_mkdir_downloads=False,
                        get_movielens_analysis=False,
                    )
                )
                df_taginfo = data_navigatingsensitivity.merge_ddd_new_names(
                    tag_csv_fname=args.dataset_dir
                    + "navigatingsensitivity_rename/ddd_rename.csv",
                    original_tag_info=df_taginfo,
                    colname_old="name_old",
                    colname_new="name_rewrite",
                )
            case _:
                raise NotImplementedError
        # remove ratings and tag links for items where metadata is missing
        log(f"{len(removed_list)} items missing data")
        if len(removed_list) > 0:
            log(f"originally {len(df_ratings)} total ratings")
            df_ratings = pd.DataFrame(
                df_ratings[~df_ratings["id_item"].isin(removed_list)]
            )
            log(f"   ... now {len(df_ratings)} total ratings")
            log(f"originally {len(df_taglinks)} tag links")
            df_taglinks = pd.DataFrame(
                df_taglinks[~df_taglinks["id_item"].isin(removed_list)]
            )
            log(f"   ... now {len(df_taglinks)} tag links")
        # we are done loading dataset
        log(f"dumping dataset anew to {fname_cache}")
        log(f"// offset dumping first: {fname_cache_offset}")
        with open(fname_cache_offset, "wb") as f:
            pickle.dump(
                (
                    df_ratings,
                    item_info,
                    removed_list,
                    df_taginfo,
                    df_taglinks,
                    args.seed,
                    args.dataset_dir,
                    args.data_user,
                    extra_args.load_tmdb,
                    args.data_tag,
                ),
                f,
            )
        log(f"// moving to main cache: {fname_cache}")
        os.rename(fname_cache_offset, fname_cache)
    maybe_pause("dataset")
except Exception as e:
    metadata["exit_code"] = 1
    with open(os.path.join(extra_args.result_dir, "crash_trace.txt"), "a") as f:
        f.write("\nDATASET STAGE CRASH\n")
        traceback.print_exc(file=f)
    dump_metadata()
    raise

# -----------------------------------------------------------------------------
# Script run: sampling
# -----------------------------------------------------------------------------

log(f"PHASE 1: sampling")

fname_cache = Path(extra_args.partial_dir_sampling) / "01_sampling.pkl"
fname_cache_offset = Path(extra_args.partial_dir_sampling) / "01_sampling.tmp.pkl"

selected_tags: list[TagID] = []
per_user_samples: list[SampleInfo] = []
# check load first
need_to_generate = True
try:
    if args.use_cache_sampling and os.path.exists(fname_cache):
        log(f"loading complete results from existing: {fname_cache}")
        with open(fname_cache, "rb") as f:
            (
                selected_tags,
                per_user_samples,
                args_seed,
                args_dataset_dir,
                args_data_user,
                eargs_load_tmdb,
                args_data_tag,
                args_sampling_steer_target_tag_ids,
                args_sampling_target_user_ids,
                args_sampling_num_users,
                args_sampling_num_users_skipn,
                args_sampling_user_min_ratings,
                args_sampling_history_size,
                args_sampling_pool_num_next,
                args_sampling_pool_num_related,
                args_sampling_pool_num_unrelated,
            ) = pickle.load(f)
        # check arg compatibility
        need_to_generate = False
        if args.seed != args_seed:
            need_to_generate = True
        elif args.dataset_dir != args_dataset_dir:
            need_to_generate = True
        elif args.data_user != args_data_user:
            need_to_generate = True
        elif extra_args.load_tmdb != eargs_load_tmdb:
            need_to_generate = True
        elif args.data_tag != args_data_tag:
            need_to_generate = True
        elif args.sampling_steer_target_tag_ids != args_sampling_steer_target_tag_ids:
            need_to_generate = True
        elif args.sampling_target_user_ids != args_sampling_target_user_ids:
            need_to_generate = True
        elif args.sampling_num_users != args_sampling_num_users:
            need_to_generate = True
        elif args.sampling_num_users_skipn != args_sampling_num_users_skipn:
            need_to_generate = True
        elif args.sampling_user_min_ratings != args_sampling_user_min_ratings:
            need_to_generate = True
        elif args.sampling_history_size != args_sampling_history_size:
            need_to_generate = True
        elif args.sampling_pool_num_next != args_sampling_pool_num_next:
            need_to_generate = True
        elif args.sampling_pool_num_related != args_sampling_pool_num_related:
            need_to_generate = True
        elif args.sampling_pool_num_unrelated != args_sampling_pool_num_unrelated:
            need_to_generate = True
except Exception as e:
    log("".join(traceback.format_exception(e)))

try:
    if not need_to_generate:
        log(f"loading worked, args agree, not generating anew")
    else:
        log(f"constructing at least partially anew")
        selected_tags, per_user_samples = sample_setup.sample_setup(
            df_taglinks=df_taglinks,
            df_ratings=df_ratings,
            item_info=item_info,
            focus_tags=(
                None
                if (len(args.sampling_steer_target_tag_ids) == 0)
                else args.sampling_steer_target_tag_ids
            ),
            tag_count=1,  # NOTE limited right now
            tag_strict_range=(0, 0.5),  # NOTE default
            tag_subprefer_range=(0.9, 1),  # NOTE default
            user_count=args.sampling_num_users,
            user_skip_n=args.sampling_num_users_skipn,
            focus_users=(
                None
                if (len(args.sampling_target_user_ids) == 0)
                else args.sampling_target_user_ids
            ),
            user_min_ratings=args.sampling_user_min_ratings,
            user_history_n=args.sampling_history_size,
            user_upcoming_n=args.sampling_pool_num_next,
            draw_num=(
                args.sampling_pool_num_related,
                args.sampling_pool_num_unrelated,
            ),
            force_exclude_items=[],  # NOTE default
            strict_separate_upcoming=True,  # NOTE default
            rng=rng_used,
            seed=args.seed,  # NOTE this is overridden by above arg
        )
        log(f"selected tags: {selected_tags}")
        log(f"sampled {len(per_user_samples)} users")
        log(f"dumping sample anew to {fname_cache}")
        log(f"// offset dumping first: {fname_cache_offset}")
        with open(fname_cache_offset, "wb") as f:
            pickle.dump(
                (
                    selected_tags,
                    per_user_samples,
                    args.seed,
                    args.dataset_dir,
                    args.data_user,
                    extra_args.load_tmdb,
                    args.data_tag,
                    args.sampling_steer_target_tag_ids,
                    args.sampling_target_user_ids,
                    args.sampling_num_users,
                    args.sampling_num_users_skipn,
                    args.sampling_user_min_ratings,
                    args.sampling_history_size,
                    args.sampling_pool_num_next,
                    args.sampling_pool_num_related,
                    args.sampling_pool_num_unrelated,
                ),
                f,
            )
        log(f"// moving to main cache: {fname_cache}")
        os.rename(fname_cache_offset, fname_cache)
    maybe_pause("sample")
except Exception as e:
    metadata["exit_code"] = 1
    with open(os.path.join(extra_args.result_dir, "crash_trace.txt"), "a") as f:
        f.write("\nSAMPLE STAGE CRASH\n")
        traceback.print_exc(file=f)
    dump_metadata()
    raise

# print output
log(f"Generated steering tasks")
log(f"Steering type: {args.sampling_steer_action}")
log(f"Steering focus tag IDs: {selected_tags}")
# write to logging csv structure
for u in per_user_samples:
    csv_export_peruser_dict[u.user_id] = {}
    csv_export_peruser_dict[u.user_id]["INFO:log_type"] = "per_user"
    csv_export_peruser_dict[u.user_id]["INFO:job_name"] = args.job_name
    csv_export_peruser_dict[u.user_id]["INFO:seed"] = args.seed
    csv_export_peruser_dict[u.user_id]["INFO:data_user"] = args.data_user
    csv_export_peruser_dict[u.user_id]["INFO:data_tag"] = args.data_tag
    csv_export_peruser_dict[u.user_id]["INFO:history_size"] = args.sampling_history_size
    csv_export_peruser_dict[u.user_id]["INFO:pool_next"] = args.sampling_pool_num_next
    csv_export_peruser_dict[u.user_id][
        "INFO:pool_related"
    ] = args.sampling_pool_num_related
    csv_export_peruser_dict[u.user_id][
        "INFO:pool_unrelated"
    ] = args.sampling_pool_num_unrelated
    csv_export_peruser_dict[u.user_id]["INFO:steer_action"] = args.sampling_steer_action
    csv_export_peruser_dict[u.user_id]["INFO:profile_method"] = args.profile_method
    csv_export_peruser_dict[u.user_id]["INFO:profile_data"] = args.profile_data
    csv_export_peruser_dict[u.user_id][
        "INFO:profile_llm_model"
    ] = args.profile_llm_model
    csv_export_peruser_dict[u.user_id]["INFO:steering_method"] = args.steering_method
    csv_export_peruser_dict[u.user_id][
        "INFO:steering_llm_model"
    ] = args.steering_llm_model
    csv_export_peruser_dict[u.user_id]["INFO:ranking_method"] = args.ranking_method
    csv_export_peruser_dict[u.user_id]["INFO:ranking_data"] = args.ranking_data
    csv_export_peruser_dict[u.user_id]["INFO:ranking_model"] = (
        args.ranking_llm_model
        if extra_args.use_llm_ranking
        else args.ranking_embed_model
    )
    csv_export_peruser_dict[u.user_id]["SAMPLE:user_id"] = u.user_id
    csv_export_peruser_dict[u.user_id]["SAMPLE:selected_tag_ids"] = selected_tags
    csv_export_peruser_dict[u.user_id]["SAMPLE:selected_tag"] = get_tag_names(
        tag_ids=selected_tags,
        df_taginfo=df_taginfo,
    )
    csv_export_peruser_dict[u.user_id]["SAMPLE:user_history_scores"] = list(
        zip(u.history_ids, u.history_scores)
    )
    csv_export_peruser_dict[u.user_id]["SAMPLE:pool"] = u.willrank_ids
    csv_export_peruser_dict[u.user_id]["SAMPLE:pool_upcoming"] = u.upcoming_ids
    csv_export_peruser_dict[u.user_id]["SAMPLE:pool_targeted"] = u.targeted_ids

# -----------------------------------------------------------------------------
# Script run: profiles
# -----------------------------------------------------------------------------

log(f"PHASE 2: profiles")

fname_cache = Path(extra_args.partial_dir_profile) / "02_profiles.pkl"
fname_cache_offset = Path(extra_args.partial_dir_profile) / "02_profiles.tmp.pkl"

partial_customized_dir = Path(extra_args.partial_dir_profile_partial)
fname_cache_partial = partial_customized_dir / "02_profiles_partial.pkl"
fname_cache_partial_offset = partial_customized_dir / "02_profiles_partial.tmp.pkl"

profiles_original: list[TextProfile] = []
addt_profiles_info: AddtNotesDict = template_addt_notes_dict()
# check load first
need_to_generate = True
try:
    if args.use_cache_profile and os.path.exists(fname_cache):
        log(f"loading complete results from existing: {fname_cache}")
        with open(fname_cache, "rb") as f:
            (
                profiles_original,
                addt_profiles_info,
                args_seed,
                args_dataset_dir,
                args_data_user,
                eargs_load_tmdb,
                args_data_tag,
                args_sampling_steer_target_tag_ids,
                args_sampling_target_user_ids,
                args_sampling_num_users,
                args_sampling_num_users_skipn,
                args_sampling_user_min_ratings,
                args_sampling_history_size,
                args_sampling_pool_num_next,
                args_sampling_pool_num_related,
                args_sampling_pool_num_unrelated,
                args_profile_method,
                args_profile_data,
                eargs_use_llm_profile,
                args_profile_llm_model,
            ) = pickle.load(f)
        # check arg compatibility
        need_to_generate = False
        if args.seed != args_seed:
            need_to_generate = True
        elif args.dataset_dir != args_dataset_dir:
            need_to_generate = True
        elif args.data_user != args_data_user:
            need_to_generate = True
        elif extra_args.load_tmdb != eargs_load_tmdb:
            need_to_generate = True
        elif args.data_tag != args_data_tag:
            need_to_generate = True
        elif args.sampling_steer_target_tag_ids != args_sampling_steer_target_tag_ids:
            need_to_generate = True
        elif args.sampling_target_user_ids != args_sampling_target_user_ids:
            need_to_generate = True
        elif args.sampling_num_users != args_sampling_num_users:
            need_to_generate = True
        elif args.sampling_num_users_skipn != args_sampling_num_users_skipn:
            need_to_generate = True
        elif args.sampling_user_min_ratings != args_sampling_user_min_ratings:
            need_to_generate = True
        elif args.sampling_history_size != args_sampling_history_size:
            need_to_generate = True
        elif args.sampling_pool_num_next != args_sampling_pool_num_next:
            need_to_generate = True
        elif args.sampling_pool_num_related != args_sampling_pool_num_related:
            need_to_generate = True
        elif args.sampling_pool_num_unrelated != args_sampling_pool_num_unrelated:
            need_to_generate = True
        elif args.profile_method != args_profile_method:
            need_to_generate = True
        elif args.profile_data != args_profile_data:
            need_to_generate = True
        elif extra_args.use_llm_profile != eargs_use_llm_profile:
            need_to_generate = True
        elif args.profile_llm_model != args_profile_llm_model:
            need_to_generate = True
except Exception as e:
    log("".join(traceback.format_exception(e)))

# load model if needed
# NOTE: this is first load of any model
try:
    if not need_to_generate:
        log(f"no need to load; already done")
    else:
        if extra_args.use_llm_profile:
            log(f"preloading llm: {args.profile_llm_model}")
            (
                preloaded_model_autolm_tokenizer,
                preloaded_model_autolm,
            ) = llm_tools.load_model_pretrained_causallm(
                model_name=args.profile_llm_model,
                token=HF_TOKEN,
                model_kwargs={
                    "device_map": "auto",
                },
            )
except Exception as e:
    log(f"model preload failed; proceeding without")
    log(f"Reason: {e}")

try:
    if not need_to_generate:
        log(f"loading worked, args agree, not generating anew")
    else:
        log(f"constructing at least partially anew")
        match args.profile_method:
            case "naive_empty":
                profiles_original, addt_profiles_info = (
                    profile_create_naive.naive_empty(
                        per_user_samples=per_user_samples,
                    )
                )
            case "naive_taglist":
                profiles_original, addt_profiles_info = (
                    profile_create_naive.naive_taglist(
                        per_user_samples=per_user_samples,
                        df_taginfo=df_taginfo,
                        df_taglinks=df_taglinks,
                        use_partial_cache=args.use_cache_profile_partial,
                        partial_cache_incr=args.use_cache_profile_partial_incr,
                        partial_cache_fname=str(fname_cache_partial),
                    )
                )
            case "list_movies":
                profiles_original, addt_profiles_info = (
                    profile_create_naive.list_movies(
                        per_user_samples=per_user_samples,
                        profile_data=args.profile_data,
                        df_ratings=df_ratings,
                        item_info=item_info,
                        df_taginfo=df_taginfo,
                        df_taglinks=df_taglinks,
                        use_partial_cache=args.use_cache_profile_partial,
                        partial_cache_incr=args.use_cache_profile_partial_incr,
                        partial_cache_fname=str(fname_cache_partial),
                    )
                )
            case "llm_movies_sentence":
                profiles_original, addt_profiles_info = (
                    profile_create_llm.llm_profile_sentence(
                        per_user_samples=per_user_samples,
                        profile_data=args.profile_data,
                        df_ratings=df_ratings,
                        item_info=item_info,
                        df_taginfo=df_taginfo,
                        df_taglinks=df_taglinks,
                        tokenizer=preloaded_model_autolm_tokenizer,
                        model=preloaded_model_autolm,
                        seed=args.seed,
                        use_partial_cache=args.use_cache_profile_partial,
                        partial_cache_incr=args.use_cache_profile_partial_incr,
                        partial_cache_fname=str(fname_cache_partial),
                    )
                )
            case "llm_movies_paragraph":
                profiles_original, addt_profiles_info = (
                    profile_create_llm.llm_profile_paragraph(
                        per_user_samples=per_user_samples,
                        profile_data=args.profile_data,
                        df_ratings=df_ratings,
                        item_info=item_info,
                        df_taginfo=df_taginfo,
                        df_taglinks=df_taglinks,
                        tokenizer=preloaded_model_autolm_tokenizer,
                        model=preloaded_model_autolm,
                        seed=args.seed,
                        use_partial_cache=args.use_cache_profile_partial,
                        partial_cache_incr=args.use_cache_profile_partial_incr,
                        partial_cache_fname=str(fname_cache_partial),
                    )
                )
            case _:
                raise NotImplementedError
        # we are done creating original profiles
        log(f"dumping profiles info anew to {fname_cache}")
        log(f"// offset dumping first: {fname_cache_offset}")
        with open(fname_cache_offset, "wb") as f:
            pickle.dump(
                (
                    profiles_original,
                    addt_profiles_info,
                    args.seed,
                    args.dataset_dir,
                    args.data_user,
                    extra_args.load_tmdb,
                    args.data_tag,
                    args.sampling_steer_target_tag_ids,
                    args.sampling_target_user_ids,
                    args.sampling_num_users,
                    args.sampling_num_users_skipn,
                    args.sampling_user_min_ratings,
                    args.sampling_history_size,
                    args.sampling_pool_num_next,
                    args.sampling_pool_num_related,
                    args.sampling_pool_num_unrelated,
                    args.profile_method,
                    args.profile_data,
                    extra_args.use_llm_profile,
                    args.profile_llm_model,
                ),
                f,
            )
        log(f"// moving to main cache: {fname_cache}")
        os.rename(fname_cache_offset, fname_cache)
    maybe_pause("profiles")
except Exception as e:
    metadata["exit_code"] = 1
    with open(os.path.join(extra_args.result_dir, "crash_trace.txt"), "a") as f:
        f.write("\nPROFILES STAGE CRASH\n")
        traceback.print_exc(file=f)
    dump_metadata()
    raise

# print output
log(f"Generated original profiles")
# write to logging csv structure
for i in range(len(per_user_samples)):
    u = per_user_samples[i]
    p = profiles_original[i]
    csv_export_peruser_dict[u.user_id]["PROFILE:profile_original"] = p
    for addt_k, addt_v in addt_profiles_info.items():
        csv_export_peruser_dict[u.user_id][f"PROFILE:ADDT:{addt_k}"] = addt_v[i]

# -----------------------------------------------------------------------------
# Script run: steering
# -----------------------------------------------------------------------------

log(f"PHASE 3: steering")

fname_cache = Path(extra_args.partial_dir_steering) / "03_steering.pkl"
fname_cache_offset = Path(extra_args.partial_dir_steering) / "03_steering.tmp.pkl"

partial_customized_dir = Path(extra_args.partial_dir_steering_partial)
fname_cache_partial = partial_customized_dir / "03_steering_partial.pkl"
fname_cache_partial_offset = partial_customized_dir / "03_steering_partial.tmp.pkl"

profiles_updated: list[TextProfile] = []
addt_steering_info: AddtNotesDict = template_addt_notes_dict()
# check load first
need_to_generate = True
try:
    if args.use_cache_steering and os.path.exists(fname_cache):
        log(f"loading complete results from existing: {fname_cache}")
        with open(fname_cache, "rb") as f:
            (
                profiles_updated,
                addt_steering_info,
                args_seed,
                args_dataset_dir,
                args_data_user,
                eargs_load_tmdb,
                args_data_tag,
                args_sampling_steer_target_tag_ids,
                args_sampling_target_user_ids,
                args_sampling_num_users,
                args_sampling_num_users_skipn,
                args_sampling_user_min_ratings,
                args_sampling_history_size,
                args_sampling_pool_num_next,
                args_sampling_pool_num_related,
                args_sampling_pool_num_unrelated,
                args_profile_method,
                args_profile_data,
                eargs_use_llm_profile,
                args_profile_llm_model,
                args_steering_method,
                args_sampling_steer_action,
                eargs_use_llm_steering,
                args_steering_llm_model,
            ) = pickle.load(f)
        # check arg compatibility
        need_to_generate = False
        if args.seed != args_seed:
            need_to_generate = True
        elif args.dataset_dir != args_dataset_dir:
            need_to_generate = True
        elif args.data_user != args_data_user:
            need_to_generate = True
        elif extra_args.load_tmdb != eargs_load_tmdb:
            need_to_generate = True
        elif args.data_tag != args_data_tag:
            need_to_generate = True
        elif args.sampling_steer_target_tag_ids != args_sampling_steer_target_tag_ids:
            need_to_generate = True
        elif args.sampling_target_user_ids != args_sampling_target_user_ids:
            need_to_generate = True
        elif args.sampling_num_users != args_sampling_num_users:
            need_to_generate = True
        elif args.sampling_num_users_skipn != args_sampling_num_users_skipn:
            need_to_generate = True
        elif args.sampling_user_min_ratings != args_sampling_user_min_ratings:
            need_to_generate = True
        elif args.sampling_history_size != args_sampling_history_size:
            need_to_generate = True
        elif args.sampling_pool_num_next != args_sampling_pool_num_next:
            need_to_generate = True
        elif args.sampling_pool_num_related != args_sampling_pool_num_related:
            need_to_generate = True
        elif args.sampling_pool_num_unrelated != args_sampling_pool_num_unrelated:
            need_to_generate = True
        elif args.profile_method != args_profile_method:
            need_to_generate = True
        elif args.profile_data != args_profile_data:
            need_to_generate = True
        elif extra_args.use_llm_profile != eargs_use_llm_profile:
            need_to_generate = True
        elif args.profile_llm_model != args_profile_llm_model:
            need_to_generate = True
        elif args.steering_method != args_steering_method:
            need_to_generate = True
        elif args.sampling_steer_action != args_sampling_steer_action:
            need_to_generate = True
        elif extra_args.use_llm_steering != eargs_use_llm_steering:
            need_to_generate = True
        elif args.steering_llm_model != args_steering_llm_model:
            need_to_generate = True
except Exception as e:
    log("".join(traceback.format_exception(e)))

# free model if needed
# NOTE: profile and steering can only use LLM right now
# so we only need to clear if the upcoming LLM is different somehow
try:
    if (preloaded_model_autolm is not None) and (
        (extra_args.use_llm_steering)
        and (args.steering_llm_model != args.profile_llm_model)
    ):
        del preloaded_model_autolm_tokenizer
        del preloaded_model_autolm
        torch.cuda.empty_cache()
        log("freed LLM weights and cleared CUDA cache before steering stage")
        preloaded_model_autolm_tokenizer = None
        preloaded_model_autolm = None
except Exception:
    log("something went wrong in freeing LLM weights")
    preloaded_model_autolm_tokenizer = None
    preloaded_model_autolm = None

# load model if needed
try:
    if not need_to_generate:
        log(f"no need to load; already done")
    else:
        if extra_args.use_llm_steering:
            if (preloaded_model_autolm is None) or (
                args.steering_llm_model != args.profile_llm_model
            ):
                log(f"preloading llm: {args.steering_llm_model}")
                (
                    preloaded_model_autolm_tokenizer,
                    preloaded_model_autolm,
                ) = llm_tools.load_model_pretrained_causallm(
                    model_name=args.steering_llm_model,
                    token=HF_TOKEN,
                    model_kwargs={
                        "device_map": "auto",
                    },
                )
except Exception as e:
    log(f"model preload failed; proceeding without")
    log(f"Reason: {e}")

try:
    if not need_to_generate:
        log(f"loading worked, args agree, not generating anew")
    else:
        log(f"constructing at least partially anew")
        match args.steering_method:
            case "naive_taglist":
                profiles_updated, addt_steering_info = profile_edit_naive.naive_taglist(
                    profiles_original=profiles_original,
                    steered_tags=selected_tags,
                    steer_action=args.sampling_steer_action,
                    df_taginfo=df_taginfo,
                    use_partial_cache=args.use_cache_steering_partial,
                    partial_cache_incr=args.use_cache_steering_partial_incr,
                    partial_cache_fname=str(fname_cache_partial),
                )
            case "append_template":
                profiles_updated, addt_steering_info = (
                    profile_edit_naive.append_template(
                        profiles_original=profiles_original,
                        steered_tags=selected_tags,
                        steer_action=args.sampling_steer_action,
                        df_taginfo=df_taginfo,
                    )
                )
            case "append_template_alt1":
                profiles_updated, addt_steering_info = (
                    profile_edit_naive.append_template(
                        profiles_original=profiles_original,
                        steered_tags=selected_tags,
                        steer_action=args.sampling_steer_action,
                        df_taginfo=df_taginfo,
                        template=profile_edit_naive.template_append_alt1,
                    )
                )
            case "append_template_alt2":
                profiles_updated, addt_steering_info = (
                    profile_edit_naive.append_template(
                        profiles_original=profiles_original,
                        steered_tags=selected_tags,
                        steer_action=args.sampling_steer_action,
                        df_taginfo=df_taginfo,
                        template=profile_edit_naive.template_append_alt2,
                    )
                )
            case "append_template_alt3_genre":
                profiles_updated, addt_steering_info = (
                    profile_edit_naive.append_template(
                        profiles_original=profiles_original,
                        steered_tags=selected_tags,
                        steer_action=args.sampling_steer_action,
                        df_taginfo=df_taginfo,
                        template=profile_edit_naive.template_append_alt3_genre,
                    )
                )
            case "append_template_alt3_ddd":
                profiles_updated, addt_steering_info = (
                    profile_edit_naive.append_template(
                        profiles_original=profiles_original,
                        steered_tags=selected_tags,
                        steer_action=args.sampling_steer_action,
                        df_taginfo=df_taginfo,
                        template=profile_edit_naive.template_append_alt3_ddd,
                    )
                )
            case "append_template_alt4":
                profiles_updated, addt_steering_info = (
                    profile_edit_naive.append_template(
                        profiles_original=profiles_original,
                        steered_tags=selected_tags,
                        steer_action=args.sampling_steer_action,
                        df_taginfo=df_taginfo,
                        template=profile_edit_naive.template_append_alt4,
                    )
                )
            case "append_template_alt5_genre":
                profiles_updated, addt_steering_info = (
                    profile_edit_naive.append_template(
                        profiles_original=profiles_original,
                        steered_tags=selected_tags,
                        steer_action=args.sampling_steer_action,
                        df_taginfo=df_taginfo,
                        template=profile_edit_naive.template_append_alt5_genre,
                    )
                )
            case "append_template_alt5_ddd":
                profiles_updated, addt_steering_info = (
                    profile_edit_naive.append_template(
                        profiles_original=profiles_original,
                        steered_tags=selected_tags,
                        steer_action=args.sampling_steer_action,
                        df_taginfo=df_taginfo,
                        template=profile_edit_naive.template_append_alt5_ddd,
                    )
                )
            case "append_template_althybrid":
                profiles_updated, addt_steering_info = (
                    profile_edit_naive.append_template(
                        profiles_original=profiles_original,
                        steered_tags=selected_tags,
                        steer_action=args.sampling_steer_action,
                        df_taginfo=df_taginfo,
                        template=profile_edit_naive.append_template_althybrid,
                    )
                )
            case "append_template_alt_emph_weak_genre":
                profiles_updated, addt_steering_info = (
                    profile_edit_naive.append_template(
                        profiles_original=profiles_original,
                        steered_tags=selected_tags,
                        steer_action=args.sampling_steer_action,
                        df_taginfo=df_taginfo,
                        template=profile_edit_naive.append_template_alt_emph_weak_genre,
                    )
                )
            case "append_template_alt_emph_strong_genre":
                profiles_updated, addt_steering_info = (
                    profile_edit_naive.append_template(
                        profiles_original=profiles_original,
                        steered_tags=selected_tags,
                        steer_action=args.sampling_steer_action,
                        df_taginfo=df_taginfo,
                        template=profile_edit_naive.append_template_alt_emph_strong_genre,
                    )
                )
            case "append_template_alt_emph_weak_ddd":
                profiles_updated, addt_steering_info = (
                    profile_edit_naive.append_template(
                        profiles_original=profiles_original,
                        steered_tags=selected_tags,
                        steer_action=args.sampling_steer_action,
                        df_taginfo=df_taginfo,
                        template=profile_edit_naive.append_template_alt_emph_weak_ddd,
                    )
                )
            case "append_template_alt_emph_strong_ddd":
                profiles_updated, addt_steering_info = (
                    profile_edit_naive.append_template(
                        profiles_original=profiles_original,
                        steered_tags=selected_tags,
                        steer_action=args.sampling_steer_action,
                        df_taginfo=df_taginfo,
                        template=profile_edit_naive.append_template_alt_emph_strong_ddd,
                    )
                )
            case "append_llm":
                profiles_updated, addt_steering_info = profile_edit_llm.append_llm(
                    profiles_original=profiles_original,
                    steered_tags=selected_tags,
                    steer_action=args.sampling_steer_action,
                    df_taginfo=df_taginfo,
                    tokenizer=preloaded_model_autolm_tokenizer,
                    model=preloaded_model_autolm,
                    seed=args.seed,
                    use_partial_cache=args.use_cache_steering_partial,
                    partial_cache_incr=args.use_cache_steering_partial_incr,
                    partial_cache_fname=str(fname_cache_partial),
                )
            case "append_llm_alt5_genre":
                profiles_updated, addt_steering_info = profile_edit_llm.append_llm(
                    profiles_original=profiles_original,
                    steered_tags=selected_tags,
                    steer_action=args.sampling_steer_action,
                    df_taginfo=df_taginfo,
                    tokenizer=preloaded_model_autolm_tokenizer,
                    model=preloaded_model_autolm,
                    seed=args.seed,
                    use_partial_cache=args.use_cache_steering_partial,
                    partial_cache_incr=args.use_cache_steering_partial_incr,
                    partial_cache_fname=str(fname_cache_partial),
                    template=profile_edit_llm.prompt_append_alt5_genre,
                )
            case "append_llm_alt5_ddd":
                profiles_updated, addt_steering_info = profile_edit_llm.append_llm(
                    profiles_original=profiles_original,
                    steered_tags=selected_tags,
                    steer_action=args.sampling_steer_action,
                    df_taginfo=df_taginfo,
                    tokenizer=preloaded_model_autolm_tokenizer,
                    model=preloaded_model_autolm,
                    seed=args.seed,
                    use_partial_cache=args.use_cache_steering_partial,
                    partial_cache_incr=args.use_cache_steering_partial_incr,
                    partial_cache_fname=str(fname_cache_partial),
                    template=profile_edit_llm.prompt_append_alt5_ddd,
                )
            case "append_llm_althybrid":
                profiles_updated, addt_steering_info = profile_edit_llm.append_llm(
                    profiles_original=profiles_original,
                    steered_tags=selected_tags,
                    steer_action=args.sampling_steer_action,
                    df_taginfo=df_taginfo,
                    tokenizer=preloaded_model_autolm_tokenizer,
                    model=preloaded_model_autolm,
                    seed=args.seed,
                    use_partial_cache=args.use_cache_steering_partial,
                    partial_cache_incr=args.use_cache_steering_partial_incr,
                    partial_cache_fname=str(fname_cache_partial),
                    template=profile_edit_llm.prompt_append_althybrid,
                )
            case "rewrite_llm":
                profiles_updated, addt_steering_info = profile_edit_llm.rewrite_llm(
                    profiles_original=profiles_original,
                    steered_tags=selected_tags,
                    steer_action=args.sampling_steer_action,
                    df_taginfo=df_taginfo,
                    tokenizer=preloaded_model_autolm_tokenizer,
                    model=preloaded_model_autolm,
                    seed=args.seed,
                    use_partial_cache=args.use_cache_steering_partial,
                    partial_cache_incr=args.use_cache_steering_partial_incr,
                    partial_cache_fname=str(fname_cache_partial),
                )
            case "rewrite_llm_alt5_genre":
                profiles_updated, addt_steering_info = profile_edit_llm.rewrite_llm(
                    profiles_original=profiles_original,
                    steered_tags=selected_tags,
                    steer_action=args.sampling_steer_action,
                    df_taginfo=df_taginfo,
                    tokenizer=preloaded_model_autolm_tokenizer,
                    model=preloaded_model_autolm,
                    seed=args.seed,
                    use_partial_cache=args.use_cache_steering_partial,
                    partial_cache_incr=args.use_cache_steering_partial_incr,
                    partial_cache_fname=str(fname_cache_partial),
                    template=profile_edit_llm.prompt_rewrite_alt5_genre,
                )
            case "rewrite_llm_alt5_ddd":
                profiles_updated, addt_steering_info = profile_edit_llm.rewrite_llm(
                    profiles_original=profiles_original,
                    steered_tags=selected_tags,
                    steer_action=args.sampling_steer_action,
                    df_taginfo=df_taginfo,
                    tokenizer=preloaded_model_autolm_tokenizer,
                    model=preloaded_model_autolm,
                    seed=args.seed,
                    use_partial_cache=args.use_cache_steering_partial,
                    partial_cache_incr=args.use_cache_steering_partial_incr,
                    partial_cache_fname=str(fname_cache_partial),
                    template=profile_edit_llm.prompt_rewrite_alt5_ddd,
                )
            case "rewrite_llm_althybrid":
                profiles_updated, addt_steering_info = profile_edit_llm.rewrite_llm(
                    profiles_original=profiles_original,
                    steered_tags=selected_tags,
                    steer_action=args.sampling_steer_action,
                    df_taginfo=df_taginfo,
                    tokenizer=preloaded_model_autolm_tokenizer,
                    model=preloaded_model_autolm,
                    seed=args.seed,
                    use_partial_cache=args.use_cache_steering_partial,
                    partial_cache_incr=args.use_cache_steering_partial_incr,
                    partial_cache_fname=str(fname_cache_partial),
                    template=profile_edit_llm.prompt_rewrite_althybrid,
                )
            case "replace":
                profiles_updated, addt_steering_info = profile_edit_naive.replace(
                    profiles_original=profiles_original,
                    steered_tags=selected_tags,
                    steer_action=args.sampling_steer_action,
                    df_taginfo=df_taginfo,
                )
            case _:
                raise NotImplementedError
        # we are done creating steered profiles
        log(f"dumping steering info anew to {fname_cache}")
        log(f"// offset dumping first: {fname_cache_offset}")
        with open(fname_cache_offset, "wb") as f:
            pickle.dump(
                (
                    profiles_updated,
                    addt_steering_info,
                    args.seed,
                    args.dataset_dir,
                    args.data_user,
                    extra_args.load_tmdb,
                    args.data_tag,
                    args.sampling_steer_target_tag_ids,
                    args.sampling_target_user_ids,
                    args.sampling_num_users,
                    args.sampling_num_users_skipn,
                    args.sampling_user_min_ratings,
                    args.sampling_history_size,
                    args.sampling_pool_num_next,
                    args.sampling_pool_num_related,
                    args.sampling_pool_num_unrelated,
                    args.profile_method,
                    args.profile_data,
                    extra_args.use_llm_profile,
                    args.profile_llm_model,
                    args.steering_method,
                    args.sampling_steer_action,
                    extra_args.use_llm_steering,
                    args.steering_llm_model,
                ),
                f,
            )
        log(f"// moving to main cache: {fname_cache}")
        os.rename(fname_cache_offset, fname_cache)
    maybe_pause("steering")
except Exception as e:
    metadata["exit_code"] = 1
    with open(os.path.join(extra_args.result_dir, "crash_trace.txt"), "a") as f:
        f.write("\nSTEERING STAGE CRASH\n")
        traceback.print_exc(file=f)
    dump_metadata()
    raise

# print output
log(f"Generated steered profiles")
# write to logging csv structure
for i in range(len(per_user_samples)):
    u = per_user_samples[i]
    p = profiles_updated[i]
    csv_export_peruser_dict[u.user_id]["STEERING:profile_steered"] = p
    for addt_k, addt_v in addt_steering_info.items():
        csv_export_peruser_dict[u.user_id][f"STEERING:ADDT:{addt_k}"] = addt_v[i]

# -----------------------------------------------------------------------------
# Script run: ranking
# -----------------------------------------------------------------------------

log(f"PHASE 4: ranking")

fname_cache = Path(extra_args.partial_dir_ranking) / "04_ranking.pkl"
fname_cache_offset = Path(extra_args.partial_dir_ranking) / "04_ranking.tmp.pkl"

partial_customized_dir = Path(extra_args.partial_dir_ranking_partial)
fname_cache_partial = partial_customized_dir / "04_ranking_partial.pkl"
fname_cache_partial_offset = partial_customized_dir / "04_ranking_partial.tmp.pkl"

rankings_all: list[RankingDF] = []
addt_ranking_info: AddtNotesDict = template_addt_notes_dict()
# check load first
need_to_generate = True
try:
    if args.use_cache_ranking and os.path.exists(fname_cache):
        log(f"loading complete results from existing: {fname_cache}")
        with open(fname_cache, "rb") as f:
            (
                rankings_all,
                addt_ranking_info,
                args_seed,
                args_dataset_dir,
                args_data_user,
                eargs_load_tmdb,
                args_data_tag,
                args_sampling_steer_target_tag_ids,
                args_sampling_target_user_ids,
                args_sampling_num_users,
                args_sampling_num_users_skipn,
                args_sampling_user_min_ratings,
                args_sampling_history_size,
                args_sampling_pool_num_next,
                args_sampling_pool_num_related,
                args_sampling_pool_num_unrelated,
                args_profile_method,
                args_profile_data,
                eargs_use_llm_profile,
                args_profile_llm_model,
                args_steering_method,
                args_sampling_steer_action,
                eargs_use_llm_steering,
                args_steering_llm_model,
                args_ranking_method,
                args_ranking_data,
                eargs_use_llm_ranking,
                args_ranking_llm_model,
                eargs_use_embed_ranking,
                args_ranking_embed_model,
            ) = pickle.load(f)
        # check arg compatibility
        need_to_generate = False
        if args.seed != args_seed:
            need_to_generate = True
        elif args.dataset_dir != args_dataset_dir:
            need_to_generate = True
        elif args.data_user != args_data_user:
            need_to_generate = True
        elif extra_args.load_tmdb != eargs_load_tmdb:
            need_to_generate = True
        elif args.data_tag != args_data_tag:
            need_to_generate = True
        elif args.sampling_steer_target_tag_ids != args_sampling_steer_target_tag_ids:
            need_to_generate = True
        elif args.sampling_target_user_ids != args_sampling_target_user_ids:
            need_to_generate = True
        elif args.sampling_num_users != args_sampling_num_users:
            need_to_generate = True
        elif args.sampling_num_users_skipn != args_sampling_num_users_skipn:
            need_to_generate = True
        elif args.sampling_user_min_ratings != args_sampling_user_min_ratings:
            need_to_generate = True
        elif args.sampling_history_size != args_sampling_history_size:
            need_to_generate = True
        elif args.sampling_pool_num_next != args_sampling_pool_num_next:
            need_to_generate = True
        elif args.sampling_pool_num_related != args_sampling_pool_num_related:
            need_to_generate = True
        elif args.sampling_pool_num_unrelated != args_sampling_pool_num_unrelated:
            need_to_generate = True
        elif args.profile_method != args_profile_method:
            need_to_generate = True
        elif args.profile_data != args_profile_data:
            need_to_generate = True
        elif extra_args.use_llm_profile != eargs_use_llm_profile:
            need_to_generate = True
        elif args.profile_llm_model != args_profile_llm_model:
            need_to_generate = True
        elif args.steering_method != args_steering_method:
            need_to_generate = True
        elif args.sampling_steer_action != args_sampling_steer_action:
            need_to_generate = True
        elif extra_args.use_llm_steering != eargs_use_llm_steering:
            need_to_generate = True
        elif args.steering_llm_model != args_steering_llm_model:
            need_to_generate = True
        elif args.ranking_method != args_ranking_method:
            need_to_generate = True
        elif args.ranking_data != args_ranking_data:
            need_to_generate = True
        elif extra_args.use_llm_ranking != eargs_use_llm_ranking:
            need_to_generate = True
        elif args.ranking_llm_model != args_ranking_llm_model:
            need_to_generate = True
        elif extra_args.use_embed_ranking != eargs_use_embed_ranking:
            need_to_generate = True
        elif args.ranking_embed_model != args_ranking_embed_model:
            need_to_generate = True
except Exception as e:
    log("".join(traceback.format_exception(e)))

# free model if needed
# NOTE: previous stages can only use the LLM
# so we only need to clear if upcoming uses embed model
# or if the upcoming LLM is different somehow
try:
    do_clear_llm = False
    if (preloaded_model_autolm is not None) and (not extra_args.use_llm_ranking):
        # upcoming is not a LLM
        do_clear_llm = True
    if (preloaded_model_autolm is not None) and (
        (extra_args.use_llm_ranking)
        and (args.ranking_llm_model != args.steering_llm_model)
    ):
        # upcoming LLM is different
        do_clear_llm = True
    if do_clear_llm:
        del preloaded_model_autolm_tokenizer
        del preloaded_model_autolm
        torch.cuda.empty_cache()
        log("freed LLM weights and cleared CUDA cache before ranking stage")
        preloaded_model_autolm_tokenizer = None
        preloaded_model_autolm = None
except Exception:
    log("something went wrong in freeing LLM weights")
    preloaded_model_autolm_tokenizer = None
    preloaded_model_autolm = None

# load model if needed
try:
    if not need_to_generate:
        log(f"no need to load; already done")
    else:
        if extra_args.use_llm_ranking:
            if (preloaded_model_autolm is None) or (
                args.ranking_llm_model != args.steering_llm_model
            ):
                log(f"preloading llm: {args.ranking_llm_model}")
                (
                    preloaded_model_autolm_tokenizer,
                    preloaded_model_autolm,
                ) = llm_tools.load_model_pretrained_causallm(
                    model_name=args.ranking_llm_model,
                    token=HF_TOKEN,
                    model_kwargs={
                        "device_map": "auto",
                    },
                )
                # preloaded_model_autolm = AutoModelForCausalLM.from_pretrained(
                #     args.ranking_llm_model,
                #     device_map="auto",
                #     token=HF_TOKEN,
                # )
        if extra_args.use_embed_ranking:
            # NOTE there should be no previous embed model, so no redundancy check
            log(f"preloading embedder: {args.ranking_embed_model}")
            preloaded_model_senttf = SentenceTransformer(args.ranking_embed_model)
except Exception as e:
    log(f"model preload failed; proceeding without")
    log(f"Reason: {e}")

try:
    if not need_to_generate:
        log(f"loading worked, args agree, not generating anew")
    else:
        log(f"constructing at least partially anew")
        # NOTE: ranking rng_used objects are used only for tiebreaking!
        # NOTE: and therefore, will not be used to ensure better tiebreaking randomness!
        match args.ranking_method:
            case "naive_random":
                rankings_all, addt_ranking_info = recommend_naive.naive_random(
                    per_user_samples=per_user_samples + per_user_samples,
                    # rng=rng_used,
                    # seed=args.seed,  # NOTE this is overridden by above arg
                )
            case "naive_tagcount":
                rankings_all, addt_ranking_info = recommend_naive.naive_tagcount(
                    per_user_samples=per_user_samples + per_user_samples,
                    profiles=profiles_original + profiles_updated,
                    df_taginfo=df_taginfo,
                    df_taglinks=df_taglinks,
                    # rng=rng_used,
                    # seed=args.seed,  # NOTE this is overridden by above arg
                    use_partial_cache=args.use_cache_ranking_partial,
                    partial_cache_incr=args.use_cache_ranking_partial_incr,
                    partial_cache_fname=str(fname_cache_partial),
                )
            case "naive_oracle":
                rankings_all, addt_ranking_info = recommend_naive.naive_oracle(
                    per_user_samples=per_user_samples + per_user_samples,
                    prioritize_accuracy=False,
                    # rng=rng_used,
                    # seed=args.seed,  # NOTE this is overridden by above arg
                    use_partial_cache=args.use_cache_ranking_partial,
                    partial_cache_incr=args.use_cache_ranking_partial_incr,
                    partial_cache_fname=str(fname_cache_partial),
                )
            case "llm_ordering":
                rankings_all, addt_ranking_info = recommend_llm.llm_ordering(
                    per_user_samples=per_user_samples + per_user_samples,
                    profiles=profiles_original + profiles_updated,
                    item_data=args.ranking_data,
                    df_ratings=df_ratings,
                    item_info=item_info,
                    df_taginfo=df_taginfo,
                    df_taglinks=df_taglinks,
                    tokenizer=preloaded_model_autolm_tokenizer,
                    model=preloaded_model_autolm,
                    targeted_tag_ids=selected_tags,
                    # rng=rng_used,
                    # seed=args.seed,  # NOTE this is overridden by above arg
                    use_partial_cache=args.use_cache_ranking_partial,
                    partial_cache_incr=args.use_cache_ranking_partial_incr,
                    partial_cache_fname=str(fname_cache_partial),
                )
            case "llm_scorepred":
                rankings_all, addt_ranking_info = recommend_llm.llm_scorepred(
                    per_user_samples=per_user_samples + per_user_samples,
                    profiles=profiles_original + profiles_updated,
                    item_data=args.ranking_data,
                    df_ratings=df_ratings,
                    item_info=item_info,
                    df_taginfo=df_taginfo,
                    df_taglinks=df_taglinks,
                    tokenizer=preloaded_model_autolm_tokenizer,
                    model=preloaded_model_autolm,
                    targeted_tag_ids=selected_tags,
                    # rng=rng_used,
                    # seed=args.seed,  # NOTE this is overridden by above arg
                    use_partial_cache=args.use_cache_ranking_partial,
                    partial_cache_incr=args.use_cache_ranking_partial_incr,
                    partial_cache_fname=str(fname_cache_partial),
                )
            case "embed_similarity":
                assert preloaded_model_senttf is not None
                rankings_all, addt_ranking_info = recommend_embed.embed_similarity(
                    per_user_samples=per_user_samples + per_user_samples,
                    profiles=profiles_original + profiles_updated,
                    item_data=args.ranking_data,
                    df_ratings=df_ratings,
                    item_info=item_info,
                    df_taginfo=df_taginfo,
                    df_taglinks=df_taglinks,
                    model=preloaded_model_senttf,
                    targeted_tag_ids=selected_tags,
                    use_partial_cache=args.use_cache_ranking_partial,
                    partial_cache_incr=args.use_cache_ranking_partial_incr,
                    partial_cache_fname=str(fname_cache_partial),
                )
            case _:
                raise NotImplementedError
        # we are done creating rankings
        log(f"dumping rankings anew to {fname_cache}")
        log(f"// offset dumping first: {fname_cache_offset}")
        with open(fname_cache_offset, "wb") as f:
            pickle.dump(
                (
                    rankings_all,
                    addt_ranking_info,
                    args.seed,
                    args.dataset_dir,
                    args.data_user,
                    extra_args.load_tmdb,
                    args.data_tag,
                    args.sampling_steer_target_tag_ids,
                    args.sampling_target_user_ids,
                    args.sampling_num_users,
                    args.sampling_num_users_skipn,
                    args.sampling_user_min_ratings,
                    args.sampling_history_size,
                    args.sampling_pool_num_next,
                    args.sampling_pool_num_related,
                    args.sampling_pool_num_unrelated,
                    args.profile_method,
                    args.profile_data,
                    extra_args.use_llm_profile,
                    args.profile_llm_model,
                    args.steering_method,
                    args.sampling_steer_action,
                    extra_args.use_llm_steering,
                    args.steering_llm_model,
                    args.ranking_method,
                    args.ranking_data,
                    extra_args.use_llm_ranking,
                    args.ranking_llm_model,
                    extra_args.use_embed_ranking,
                    args.ranking_embed_model,
                ),
                f,
            )
        log(f"// moving to main cache: {fname_cache}")
        os.rename(fname_cache_offset, fname_cache)
    maybe_pause("ranking")
except Exception as e:
    metadata["exit_code"] = 1
    with open(os.path.join(extra_args.result_dir, "crash_trace.txt"), "a") as f:
        f.write("\nRANKING STAGE CRASH\n")
        traceback.print_exc(file=f)
    dump_metadata()
    raise


def augment_ranking(
    r: RankingDF,
    u: SampleInfo,
) -> RankingDF:
    ranking = r.copy()
    # add relevance column to ranking
    ranking["relevance"] = ranking["id_item"].apply(
        lambda x: 1 if (x in u.targeted_ids) else 0
    )
    # add upcoming column to ranking
    ranking["upcoming"] = ranking["id_item"].apply(
        lambda x: 1 if (x in u.upcoming_ids) else 0
    )
    # add popularity column to ranking
    ranking["popularity"] = ranking["id_item"].apply(
        lambda x: getattr(item_info[x], "popularity", None)
    )
    # reset index
    ranking = ranking.sort_values(
        by="score", ascending=False, na_position="last"
    ).reset_index(drop=True)
    return ranking


# print output
log(f"Generated all rankings")
# write to logging csv structure
for i in range(len(per_user_samples)):
    u = per_user_samples[i]
    r_old = rankings_all[i]
    r_new = rankings_all[i + len(per_user_samples)]
    csv_export_peruser_dict[u.user_id]["RANKING:ranking_original"] = r_old
    csv_export_peruser_dict[u.user_id]["RANKING:ranking_steered"] = r_new
    csv_export_peruser_dict[u.user_id]["RANKING:ranking_original_augm"] = (
        augment_ranking(
            r_old,
            u,
        )
    )
    csv_export_peruser_dict[u.user_id]["RANKING:ranking_steered_augm"] = (
        augment_ranking(
            r_new,
            u,
        )
    )
    for addt_k, addt_v in addt_ranking_info.items():
        csv_export_peruser_dict[u.user_id][f"RANKING:ADDT:ORIGN:{addt_k}"] = addt_v[
            : len(per_user_samples)
        ][i]
        csv_export_peruser_dict[u.user_id][f"RANKING:ADDT:STEER:{addt_k}"] = addt_v[
            len(per_user_samples) :
        ][i]

# -----------------------------------------------------------------------------
# Script run: scoring
# -----------------------------------------------------------------------------

log(f"PHASE 5: scoring")

# construct the list of scores to run
# tuple of final name, acc/tag, metric type name, k
total_num_items: int = args.sampling_pool_num_related + args.sampling_pool_num_unrelated
will_compute_scores: list[Tuple[str, str, str, int]] = []
for items in ["acc", "tag"]:
    # minrank: only full
    will_compute_scores += [
        (f"{items}_minrank@full", items, "minrank", total_num_items)
    ]
    # precision: all k values
    will_compute_scores += [
        (f"{items}_precision@{k}", items, "precision", k) for k in args.eval_k
    ]
    # recall: all k values
    will_compute_scores += [
        (f"{items}_recall@{k}", items, "recall", k) for k in args.eval_k
    ]
    # fscore: all k values
    will_compute_scores += [
        (f"{items}_fscore@{k}", items, "fscore", k) for k in args.eval_k
    ]
    # mrr: all k values, and full
    will_compute_scores += [(f"{items}_mrr@{k}", items, "mrr", k) for k in args.eval_k]
    will_compute_scores += [(f"{items}_mrr@full", items, "mrr", total_num_items)]
    # ndcg: all k values, and full
    will_compute_scores += [
        (f"{items}_ndcg@{k}", items, "ndcg", k) for k in args.eval_k
    ]
    will_compute_scores += [(f"{items}_ndcg@full", items, "ndcg", total_num_items)]
    # auc: only full
    will_compute_scores += [(f"{items}_auc@full", items, "auc", total_num_items)]


# evaluate metric per metric type...
def evaluate_metric(
    ranking: RankingDF,
    targeted_ids: list[ItemID],
    metric_type: str,
    metric_k: int,
) -> float:
    match metric_type:
        case "minrank":
            return metrics.minrank(
                ranking=ranking,
                targeted_ids=targeted_ids,
                k=metric_k,
            )
        case "precision":
            return metrics.precision(
                ranking=ranking,
                targeted_ids=targeted_ids,
                k=metric_k,
            )
        case "recall":
            return metrics.recall(
                ranking=ranking,
                targeted_ids=targeted_ids,
                k=metric_k,
            )
        case "fscore":
            return metrics.fscore(
                ranking=ranking,
                targeted_ids=targeted_ids,
                k=metric_k,
                beta=1,  # NOTE default beta
            )
        case "mrr":
            return metrics.mrr(
                ranking=ranking,
                targeted_ids=targeted_ids,
                k=metric_k,
            )
        case "ndcg":
            return metrics.ndcg(
                ranking=ranking,
                targeted_ids=targeted_ids,
                k=metric_k,
            )
        case "auc":
            return metrics.auc(
                ranking=ranking,
                targeted_ids=targeted_ids,
                k=metric_k,
            )
        case _:
            raise NotImplementedError


def pick_target(
    s_type: str,
    user_sample: SampleInfo,
) -> list[ItemID]:
    match s_type:
        case "acc":
            return user_sample.upcoming_ids
        case "tag":
            return user_sample.targeted_ids
        case _:
            raise NotImplementedError


scores_old: dict[str, list[float]] = {}
scores_new: dict[str, list[float]] = {}
scores_diff: dict[str, list[float]] = {}
scores_diff_avg: dict[str, float] = {}
try:
    for s_name, s_items, s_type, s_k in will_compute_scores:
        assert len(rankings_all) == 2 * len(per_user_samples)
        rankings_old = rankings_all[: len(per_user_samples)]
        rankings_new = rankings_all[len(per_user_samples) :]
        scores_old[s_name] = [
            evaluate_metric(
                ranking=r,
                targeted_ids=pick_target(s_items, u),
                metric_type=s_type,
                metric_k=s_k,
            )
            for r, u in zip(rankings_old, per_user_samples)
        ]
        scores_new[s_name] = [
            evaluate_metric(
                ranking=r,
                targeted_ids=pick_target(s_items, u),
                metric_type=s_type,
                metric_k=s_k,
            )
            for r, u in zip(rankings_new, per_user_samples)
        ]
    assert len(scores_old) == len(will_compute_scores)
    assert len(scores_new) == len(will_compute_scores)
    scores_diff = {
        s_name: [(n - o) for o, n in zip(scores_old[s_name], scores_new[s_name])]
        for s_name in scores_old.keys()
    }
    scores_diff_avg = {
        s_name: (
            (sum(scores_diff[s_name]) / len(scores_diff[s_name]))
            if (len(scores_diff[s_name]) > 0)
            else (np.nan)
        )
        for s_name in scores_diff
    }
except Exception as e:
    metadata["exit_code"] = 1
    with open(os.path.join(extra_args.result_dir, "crash_trace.txt"), "a") as f:
        f.write("\nSCORING STAGE CRASH\n")
        traceback.print_exc(file=f)
    dump_metadata()
    raise

# print output
log(f"Generated all scores")
# log(f"Pre-steering scores:")
# log(json.dumps(scores_old, indent=2))
# log(f"Post-steering scores:")
# log(json.dumps(scores_new, indent=2))
# log(f"Diff scores:")
# log(json.dumps(scores_diff, indent=2))
log(f"Average score diffs:")
log(json.dumps(scores_diff_avg, indent=2))
# write to logging csv structure
for i in range(len(per_user_samples)):
    u = per_user_samples[i]
    for s_name in scores_diff.keys():
        csv_export_peruser_dict[u.user_id][f"SCORES:ORIGN:{s_name}"] = scores_old[
            s_name
        ][i]
        csv_export_peruser_dict[u.user_id][f"SCORES:STEER:{s_name}"] = scores_new[
            s_name
        ][i]
        csv_export_peruser_dict[u.user_id][f"SCORES:DELTA:{s_name}"] = scores_diff[
            s_name
        ][i]

# -----------------------------------------------------------------------------
# Script run: final output
# -----------------------------------------------------------------------------

fname_csv_export = os.path.join(extra_args.result_dir, "results.csv")

log(f"dumping final CSV export to {fname_csv_export}")

# convert and write csv
new_csv_export_peruser_dict = {
    k: {
        i_k: i_v
        # i_k: pprint.pformat(i_v, indent=0, sort_dicts=False,)
        for i_k, i_v in v.items()
    }
    for k, v in csv_export_peruser_dict.items()
}
with pd.option_context(
    "display.max_rows",
    None,
    "display.max_columns",
    None,
):
    pd.DataFrame.from_dict(
        new_csv_export_peruser_dict,
        orient="index",
    ).to_csv(
        fname_csv_export,
        index=False,
    )

metadata["end_ts"] = datetime.now().isoformat(timespec="seconds")
dump_metadata()
log("DONE")

# # --- accumulated info
# csv_export_peruser_dict  # csv information
# preloaded_model_autolm_tokenizer  # tokenizer maybe loaded
# preloaded_model_autolm  # model maybe loaded
# preloaded_model_senttf  # model maybe loaded
# # --- dataset
# df_ratings  # rating data
# item_info  # item metadata
# removed_list  # items that were removed for having no metadata
# df_taginfo  # tag metadata
# df_taglinks  # tags connected
# # --- sampling
# selected_tags  # tag(s) to steer
# per_user_samples  # ordered list of experiment setup per user
# # --- profiles
# profiles_original
# addt_profiles_info
# # --- steering
# profiles_updated
# addt_steering_info
# # --- ranking
# rankings_all
# addt_ranking_info
# # --- scoring
# scores_old
# scores_new
# scores_diff
# scores_diff_avg
# # --- end
