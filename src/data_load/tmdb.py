from dataclasses import dataclass
from pathlib import Path
import pickle
import pprint
from typing import Any, Dict, List, Tuple

from custom_types import ItemInfo, ItemID
from data_load.movielens import MovieLensItemInfoDict


@dataclass
class TmdbItemInfo(ItemInfo):
    title: str
    release_year: int
    description: str
    popularity: float
    genres: List[str]


TmdbItemInfoDict = Dict[ItemID, TmdbItemInfo]


def merge_tmdb_into_movielens(
    tmdb_pkl_fname: str,
    original_item_info: MovieLensItemInfoDict,
) -> Tuple[
    TmdbItemInfoDict,
    List[ItemID],
]:
    """
    Augment existing Movielens item information with TMDB information, joined by item ID

    :param tmdb_pkl_fname: Filename for TMDB data to use
    :type tmdb_pkl_fname: str
    :param original_item_info: Original Movielens data
    :type original_item_info: MovieLensItemInfoDict
    :return: Tuple of augmented item information, list of removed item IDs (due to being missing from TMDB dataset)
    :rtype: Tuple[TmdbItemInfoDict, List[ItemID]]
    """
    with open(tmdb_pkl_fname, "rb") as f:
        tmdb_data = pickle.load(f)
    results = {}
    removed_list = []
    for movielens_id, original_data in original_item_info.items():
        try:
            new_data = tmdb_data[movielens_id]
            release_date = new_data["info"]["release_date"].split(
                "-"
            )  # formatted like "1999-03-31"
            assert len(release_date) == 3
            results[movielens_id] = TmdbItemInfo(
                **{
                    "title": new_data["info"]["title"],
                    "release_year": int(release_date[0]),
                    "description": new_data["info"]["overview"],
                    "popularity": new_data["info"]["popularity"],
                    "genres": original_data.genres,
                }
            )
        except:
            removed_list.append(movielens_id)
    return results, removed_list
