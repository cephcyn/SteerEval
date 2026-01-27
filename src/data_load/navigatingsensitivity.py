# Heavily borrowed from https://github.com/sdean-group/Navigating-Sensitivity/blob/main/ml-ddd/ml-ddd_data_generation.py

import os
import pprint
from typing import Any, Optional, Tuple
import pathlib
from pathlib import Path
import tempfile
from zipfile import ZipFile
import shutil
import pickle

import requests
import pandas as pd
from huggingface_hub import hf_hub_download

from custom_types import TagInfoDF, validate_tag_info_df
from custom_types import TagLinksDF, validate_tag_links_df


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"


def download_movielens(
    input_data_dir: Path,
    mkdir=True,
    verbose=False,
) -> None:
    url = MOVIELENS_URL
    if verbose is True:
        print(f"Downloading from {url}")
    output_dir = pathlib.Path(input_data_dir).resolve()
    if not output_dir.exists():
        if mkdir:
            output_dir.mkdir(exist_ok=True)
        else:
            raise Exception(f"{output_dir} does not exist. Pass `mkdir=True`")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get("content-length", 0))
        with tempfile.NamedTemporaryFile(mode="rb+") as temp_f:
            downloaded = 0
            dl_iteration = 0
            chunk_size = 8192
            total_chunks = (
                total_size_in_bytes / chunk_size if total_size_in_bytes else 100
            )
            for chunk in r.iter_content(chunk_size=chunk_size):
                if verbose is True:
                    downloaded += chunk_size
                    dl_iteration += 1
                    percent = 100 * dl_iteration * 1.0 / total_chunks
                    if dl_iteration % 10 == 0 and percent < 100:
                        print(f"Completed {percent:2f}%")
                    elif percent >= 99.9:
                        print(f"Download completed. Now unzipping...")
                temp_f.write(chunk)
            with ZipFile(temp_f, "r") as zipf:
                zipf.extractall(output_dir)
                if verbose is True:
                    print(
                        f"\n\nUnzipped.\n\nFiles downloaded and unziped to:\n\n{input_data_dir.resolve()}"
                    )


def download_ml_ratings(
    input_data_dir: Path,
) -> pd.DataFrame:
    print("DOWNLOADING MOVIELENS 25M...")

    download_movielens(input_data_dir)

    ratings_path = os.path.join(os.path.join(input_data_dir, "ml-25m"), "ratings.csv")
    ratings_df = pd.read_csv(ratings_path)

    print("COMPLETED DOWNLOAD")
    return ratings_df


def download_ddd_warnings(
    input_data_dir: Path,
) -> dict:
    print("DOWNLOADING DOES THE DOG DIE WARNING DICTIONARY...")
    repo_id = "sdeangroup/NavigatingSensitivity"
    ddd_dict = "ddd_dict.pkl"

    ddd_path = open(
        hf_hub_download(repo_id=repo_id, filename=ddd_dict, repo_type="dataset"), "rb"
    )
    ddd_destination = open(f"{input_data_dir}/ddd_dict.pkl", "wb")
    shutil.copyfileobj(ddd_path, ddd_destination)

    with open(f"{input_data_dir}/ddd_dict.pkl", "rb") as handle:
        ddd_dict = pickle.load(handle)

    print("COMPLETED DOWNLOAD")
    return ddd_dict


def get_warning_votes(
    votes: dict[str, int],
) -> Tuple[int, int, int, int]:
    """
    Takes votes as a dict shaped like {"yesSum": a, "noSum": b}
    Returns 1-hot list: clear yes (yesSum > 75% total), clear no (noSum > 75% total), unclear (yesSum < 75% and noSum < 75%), no votes (total = 0)
    """
    majority = 0.75 * (votes["yesSum"] + votes["noSum"])
    if majority == 0:
        # no votes
        return 0, 0, 0, 1
    elif votes["yesSum"] > majority:
        # majority yes votes
        return 1, 0, 0, 0
    elif votes["noSum"] > majority:
        # majority no votes
        return 0, 1, 0, 0
    else:
        # no clear consensus
        return 0, 0, 1, 0


def get_sensitivity_table(
    ddd_dict: dict,
) -> pd.DataFrame:
    """
    Construct sensitivity df based on NavigatingSensitivity dict
    """
    data = {}
    warnings = set()
    for warning, work_votes in ddd_dict.items():
        if warning not in warnings:
            warnings.add(warning)
        for work_id, votes in work_votes.items():
            if work_id not in data:
                data[work_id] = {}
            (
                data[work_id][f"Clear Yes: {warning}"],
                data[work_id][f"Clear No: {warning}"],
                data[work_id][f"Unclear: {warning}"],
                data[work_id][f"No Votes: {warning}"],
            ) = get_warning_votes(votes)

    # Creating DataFrame
    sensitivity_table = pd.DataFrame(data).T.fillna(0).astype(int)
    sensitivity_table.reset_index(inplace=True)
    sensitivity_table.rename(columns={"index": "work_id"}, inplace=True)

    return sensitivity_table


def filter_tables(
    sensitivity_table: pd.DataFrame,
    interaction_table: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter the dataframes to only contain users with at least 3 interactions and works with at least 3 interactions.
    """
    interaction_table = interaction_table.rename(
        columns={"userId": "user_id", "movieId": "work_id"}
    )
    print(
        f"Initial number of users before filtering: {len(interaction_table["user_id"].unique())}"
    )
    print(f"Initial number of works before filtering: {len(sensitivity_table)}")

    for i in range(3):
        # Remove users from interaction_table with less than three interactions
        user_counts = interaction_table["user_id"].value_counts()
        users_to_keep = user_counts[user_counts >= 3].index
        interaction_table = interaction_table[
            interaction_table["user_id"].isin(users_to_keep)
        ]
        existing_users = set(interaction_table["user_id"].unique())
        print(
            f"Number of users with at least 3 interactions (pass {i}): {len(existing_users)}"
        )

        # Remove works that appear less than three times in the interaction_table
        interactions = interaction_table["work_id"].value_counts()
        works_to_keep = interactions[interactions >= 3].index
        sensitivity_table = sensitivity_table[
            sensitivity_table["work_id"].isin(works_to_keep)
        ]
        print(
            f"Number of works with at least 3 likes (pass {i}): {len(sensitivity_table)}"
        )

        # Update interaction_table to include only works that are present after filtering users and works
        works_to_keep = sensitivity_table["work_id"].unique()
        interaction_table = interaction_table[
            interaction_table["work_id"].isin(works_to_keep)
        ]

    print(f"Final number of users: {len(interaction_table["user_id"].unique())}")
    print(f"Final number of works: {len(sensitivity_table)}")
    print(f"Final number of interactions: {len(interaction_table)}")

    return sensitivity_table, interaction_table


def add_summary_stats(
    sensitivity_table: pd.DataFrame,
    interaction_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Creates and returns an updated version of sensitivity_table
    """
    # Calculate the number of ratings (n_ratings) and average rating (av_rating) for each work
    work_ratings_summary = (
        interaction_table.groupby("work_id")
        .agg(n_ratings=("rating", "count"), av_rating=("rating", "mean"))
        .reset_index()
    )

    sensitivity_table = pd.merge(
        sensitivity_table, work_ratings_summary, on="work_id", how="left"
    )
    user_dict = (
        interaction_table.groupby("work_id")
        .apply(
            lambda x: {user: rating for user, rating in zip(x["user_id"], x["rating"])}
        )
        .to_dict()
    )

    def get_ratings(work_id):
        return user_dict.get(work_id, {})

    # Add dictionary of each user and their rating for each each
    sensitivity_table["user_ratings"] = sensitivity_table["work_id"].map(get_ratings)

    return sensitivity_table


def navigatingsensitivity_tags_ddd(
    raw_dir_path: Path,
    out_dir_path: Optional[Path] = None,
    allow_mkdir_downloads: bool = False,
    get_movielens_analysis: bool = False,
    **_: Any,
) -> Tuple[
    TagInfoDF,
    TagLinksDF,
]:
    """
    Load DoesTheDogDie trigger warning tag dataset, based on NavigatingSensitivity source

    :param raw_dir_path: Directory to read NavigatingSensitivity files from
    :type raw_dir_path: Path
    :param out_dir_path: Directory to write NavigatingSensitivity interaction analysis to
    :type out_dir_path: Optional[Path]
    :param allow_mkdir_downloads: Enable MovieLens or NavigatingSensitivity file downloads, if needed
    :type allow_mkdir_downloads: bool
    :param get_movielens_analysis: Perform interaction analysis, if needed
    :type get_movielens_analysis: bool
    :param _: Overflow args
    :type _: Any
    :return: Tuple of tag and tag-link information
    :rtype: Tuple[TagInfoDF, TagLinksDF]
    """
    pprint.pprint(f"Doing MovieLens comparison analysis?: {get_movielens_analysis}")
    # do NavigatingSensitivity stage loading...
    # retrieve/load movielens and ddd files
    if allow_mkdir_downloads:
        pprint.pprint("downloading new DDD file")
        ddd_dict = download_ddd_warnings(input_data_dir=raw_dir_path)
    else:
        pprint.pprint("loading existing DDD file")
        with open(f"{raw_dir_path}/ddd_dict.pkl", "rb") as f:
            ddd_dict = pickle.load(f)
    # reformat
    if get_movielens_analysis:
        if allow_mkdir_downloads:
            ratings_df = download_ml_ratings(input_data_dir=raw_dir_path)
        else:
            ratings_df = pd.read_csv(
                os.path.join(os.path.join(raw_dir_path, "ml-25m"), "ratings.csv")
            )
        sensitivity_table = get_sensitivity_table(ddd_dict)
        sensitivity_table, interaction_table = filter_tables(
            sensitivity_table, ratings_df
        )
        sensitivity_table = add_summary_stats(sensitivity_table, interaction_table)
        # output NavigatingSensitivity reformatted CSVs
        pprint.pprint(sensitivity_table)
        pprint.pprint(interaction_table)
        if out_dir_path is not None:
            sensitivity_table.to_csv(
                os.path.join(out_dir_path, "ml-ddd_sensitivity_table.csv"),
                index=False,
            )
            interaction_table.to_csv(
                os.path.join(out_dir_path, "ml-ddd_interaction_table.csv"),
                index=False,
            )

    # convert to taginfo and taglinks
    dict_taginfo = {
        "id_tag": [],
        "name": [],
    }
    dict_taglinks = {
        "id_item": [],
        "id_tag": [],
        "relevant": [],
        "score_relevant": [],
        "score_known": [],
        "votes_yes": [],  # custom
        "votes_no": [],  # custom
        # Add any other custom relevance info here...
    }
    mapping_tags = {}
    for tag_name, tag_votes in ddd_dict.items():
        if tag_name not in mapping_tags:
            tag_id = len(mapping_tags)
            mapping_tags[tag_name] = tag_id
            dict_taginfo["id_tag"].append(tag_id)
            dict_taginfo["name"].append(tag_name)
        tag_id = mapping_tags[tag_name]
        for work_id, votes in tag_votes.items():
            total_votes = votes["yesSum"] + votes["noSum"]
            is_relevant = False
            if total_votes > 0:
                is_relevant = votes["yesSum"] > 0.75 * total_votes
            score_relevant = (
                (votes["yesSum"] / total_votes) if (total_votes > 0) else None
            )
            dict_taglinks["id_item"].append(work_id)
            dict_taglinks["id_tag"].append(tag_id)
            dict_taglinks["relevant"].append(is_relevant)
            dict_taglinks["score_relevant"].append(score_relevant)
            dict_taglinks["score_known"].append(min(1, total_votes / 10))
            dict_taglinks["votes_yes"].append(votes["yesSum"])
            dict_taglinks["votes_no"].append(votes["noSum"])
    df_taginfo = pd.DataFrame(dict_taginfo)
    df_taglinks = pd.DataFrame(dict_taglinks)

    # verify that output matches format minimum
    validate_tag_info_df(df_taginfo)
    validate_tag_links_df(df_taglinks)
    # return results
    return df_taginfo, df_taglinks


def merge_ddd_new_names(
    tag_csv_fname: str,
    original_tag_info: TagInfoDF,
    colname_old: str = "name_old",
    colname_new: str = "name_rewrite",
) -> TagInfoDF:
    """
    Replace tag names from an existing TagInfoDF

    :param tag_csv_fname: fname to pull new tag names from
    :type tag_csv_fname: str
    :param original_tag_info: old tag information
    :type original_tag_info: TagInfoDF
    :param colname_old: Column name for old tag name
    :type colname_old: str
    :param colname_new: Column name for new tag name
    :type colname_new: str
    :return: Updated tag information
    :rtype: TagInfoDF
    """
    df_csv = pd.read_csv(tag_csv_fname)
    # build remapping dict
    csv_remap = {}
    for ix, row in df_csv.iterrows():
        csv_remap[row[colname_old].lower()] = row[colname_new]
    # apply column name remap
    new_tag_info = original_tag_info.copy()
    new_tag_info["name"] = new_tag_info["name"].apply(
        lambda x: (csv_remap[x.lower()]) if (x.lower() in csv_remap) else (x)
    )
    return new_tag_info
