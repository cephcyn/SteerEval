# Heavily borrowed from https://github.com/sdean-group/Navigating-Sensitivity/blob/main/ml-ddd/ml-ddd_data_generation.py

import os
import pprint
from typing import Tuple, Dict, List, Any
from dataclasses import dataclass
import pathlib
from pathlib import Path
import tempfile
from zipfile import ZipFile

import requests
import pandas as pd

from custom_types import ItemInfo, ItemID
from custom_types import RatingDataDF, validate_rating_data_df
from custom_types import TagInfoDF, validate_tag_info_df
from custom_types import TagLinksDF, validate_tag_links_df


@dataclass
class MovieLensItemInfo(ItemInfo):
    title: str
    genres: List[str]


MovieLensItemInfoDict = Dict[ItemID, MovieLensItemInfo]

# ===

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"


def download_movielens_25m(
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


def download_ml25m_ratings(
    input_data_dir: Path,
) -> pd.DataFrame:
    print("DOWNLOADING MOVIELENS 25M...")

    download_movielens_25m(input_data_dir=input_data_dir)

    ratings_path = os.path.join(os.path.join(input_data_dir, "ml-25m"), "ratings.csv")
    ratings_df = pd.read_csv(ratings_path)

    print("COMPLETED DOWNLOAD")
    return ratings_df


def movielens_25m_ratings_items(
    raw_dir_path: Path,
    allow_mkdir_downloads: bool,
    **_: Any,
) -> Tuple[
    RatingDataDF,
    MovieLensItemInfoDict,
]:
    """
    Load Movielens 25M item metadata and ratings datasets.

    :param raw_dir_path: Directory to read Movielens 25M files from
    :type raw_dir_path: Path
    :param allow_mkdir_downloads: Enable MovieLens file download, if needed
    :type allow_mkdir_downloads: bool
    :param _: Overflow args
    :type _: Any
    :return: Tuple of rating data, movie item metadata information
    :rtype: Tuple[RatingDataDF, MovieLensItemInfoDict]
    """
    # retrieve/load movielens files
    if allow_mkdir_downloads:
        df_ratings = download_ml25m_ratings(input_data_dir=raw_dir_path)
    else:
        df_ratings = pd.read_csv(
            os.path.join(os.path.join(raw_dir_path, "ml-25m"), "ratings.csv")
        )

    # reformat df_ratings
    df_ratings = df_ratings.rename(
        columns={
            "userId": "id_user",
            "movieId": "id_item",
            "rating": "score",
            "timestamp": "time",
        }
    )

    # get item data
    df_items = pd.read_csv(
        os.path.join(os.path.join(raw_dir_path, "ml-25m"), "movies.csv")
    )
    itemdict = {
        int(r["movieId"]): MovieLensItemInfo(
            **{
                "title": r["title"],
                "genres": [
                    e for e in r["genres"].split("|") if e != "(no genres listed)"
                ],
            }
        )
        for _, r in df_items.iterrows()
    }

    # verify that output matches format minimum
    validate_rating_data_df(df_ratings)
    # return results
    return df_ratings, itemdict


def movielens_tag_genres(
    raw_dir_path: Path,
    **_: Any,
) -> Tuple[
    TagInfoDF,
    TagLinksDF,
]:
    """
    Load Movielens 25M genre tag dataset

    :param raw_dir_path: Directory to read Movielens 25M files from
    :type raw_dir_path: Path
    :param _: Overflow args
    :type _: Any
    :return: Tuple of tag and tag-link information
    :rtype: Tuple[TagInfoDF, TagLinksDF]
    """
    # assume that movielens does not need to be (newly) downloaded
    # since we evidently already use the dataset
    # load relevant movielens files
    df_items = pd.read_csv(
        os.path.join(os.path.join(raw_dir_path, "ml-25m"), "movies.csv")
    )

    # do some custom reformatting of genre information
    # start by building the collection of genres
    df_items["genres_raw_split"] = df_items["genres"].map(
        lambda x: [e for e in x.split("|") if e != "(no genres listed)"]
    )
    all_genres_set = set()
    for g_list in df_items["genres_raw_split"]:
        for g in g_list:
            if g not in all_genres_set:
                all_genres_set.add(g)
    all_genres_sorted = sorted(list(all_genres_set))
    genre_dict = {v: k for k, v in enumerate(all_genres_sorted)}
    # now build a full relevance data structure
    mapping_tagdata = {}
    for i, r in df_items.iterrows():
        mapping_tagdata[r["movieId"]] = [
            (g, g in r["genres_raw_split"]) for g in genre_dict
        ]

    # convert mappings to taginfo and taglinks
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
        # Add any other custom relevance info here...
    }
    for genre_name, genre_id in genre_dict.items():
        dict_taginfo["id_tag"].append(genre_id)
        dict_taginfo["name"].append(genre_name)
    for movie_id, tag_infos in mapping_tagdata.items():
        for genre_name, genre_present in tag_infos:
            dict_taglinks["id_item"].append(movie_id)
            dict_taglinks["id_tag"].append(genre_dict[genre_name])
            dict_taglinks["relevant"].append(genre_present)
            dict_taglinks["score_relevant"].append(1 if genre_present else 0)
            dict_taglinks["score_known"].append(1)
    df_taginfo = pd.DataFrame(dict_taginfo)
    df_taglinks = pd.DataFrame(dict_taglinks)

    # verify that output matches format minimum
    validate_tag_info_df(df_taginfo)
    validate_tag_links_df(df_taglinks)
    # return results
    return df_taginfo, df_taglinks


def movielens_tag_genome(
    raw_dir_path: Path,
    **_: Any,
) -> Tuple[
    TagInfoDF,
    TagLinksDF,
]:
    """
    Load Movielens 25M genome tag dataset

    :param raw_dir_path: Directory to read Movielens 25M files from
    :type raw_dir_path: Path
    :param _: Overflow args
    :type _: Any
    :return: Tuple of tag and tag-link information
    :rtype: Tuple[TagInfoDF, TagLinksDF]
    """
    # assume that movielens does not need to be (newly) downloaded
    # since we evidently already use the dataset
    # load relevant movielens files
    df_genome_tags = pd.read_csv(
        os.path.join(os.path.join(raw_dir_path, "ml-25m"), "genome-tags.csv")
    )
    df_genome_scores = pd.read_csv(
        os.path.join(os.path.join(raw_dir_path, "ml-25m"), "genome-scores.csv")
    )

    # misc exploration of relevance distribution
    # bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # pprint.pprint(pd.cut(df_genome_scores["relevance"], bins).value_counts())

    # convert to taginfo and taglinks
    # dict_taginfo = {
    #     "id_tag": [],
    #     "name": [],
    # }
    # dict_taglinks = {
    #     "id_item": [],
    #     "id_tag": [],
    #     "relevant": [],
    #     "score_relevant": [],
    #     "score_known": [],
    #     # Add any other custom relevance info here...
    # }
    df_genome_tags["id_tag"] = df_genome_tags["tagId"]
    df_genome_tags["name"] = df_genome_tags["tag"]
    df_genome_scores["id_item"] = df_genome_scores["movieId"]
    df_genome_scores["id_tag"] = df_genome_scores["tagId"]
    df_genome_scores["relevant"] = df_genome_scores["relevance"].map(lambda x: x > 0.5)
    df_genome_scores["score_relevant"] = df_genome_scores["relevance"]
    df_genome_scores["score_known"] = df_genome_scores["relevance"].map(lambda x: 1)
    df_taginfo = pd.DataFrame(
        df_genome_tags[
            [
                "id_tag",
                "name",
            ]
        ]
    )
    df_taglinks = pd.DataFrame(
        df_genome_scores[
            [
                "id_item",
                "id_tag",
                "relevant",
                "score_relevant",
                "score_known",
            ]
        ]
    )

    # verify that output matches format minimum
    validate_tag_info_df(df_taginfo)
    validate_tag_links_df(df_taglinks)
    # return results
    return df_taginfo, df_taglinks
