from dataclasses import asdict, dataclass
import pprint
from typing import Any, Dict, List, Mapping

import pandas as pd

# IDs
UserID = int
ItemID = int
TagID = int

# Rating dataset format
RatingDataDF = pd.DataFrame


def validate_rating_data_df(
    df: RatingDataDF,
) -> None:
    required = {
        "id_user",  # int, UserID
        "id_item",  # int, ItemID
        "score",  # float
        "time",  # int
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def template_rating_data_df() -> RatingDataDF:
    return pd.DataFrame(
        {
            "id_user": [],
            "id_item": [],
            "score": [],
            "time": [],
        }
    )


# Item metadata format


@dataclass
class ItemInfo:
    """
    Data class containing all fields for item metadata, with dict converter
    Can construct from a dict by calling ItemData(**d)
    """

    def __str__(self) -> str:
        return pprint.pformat(
            asdict(self),
            indent=2,
        )

    def __getitem__(self, key) -> Any:
        return getattr(self, key)


ItemInfoDict = Mapping[ItemID, ItemInfo]


def template_item_info_dict() -> ItemInfoDict:
    return {}


# Tag metadata format
TagInfoDF = pd.DataFrame


def validate_tag_info_df(
    df: TagInfoDF,
) -> None:
    required = {
        "id_tag",  # int, TagID
        "name",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def template_tag_info_df() -> TagInfoDF:
    return pd.DataFrame(
        {
            "id_tag": [],
            "name": [],
        }
    )


# Tag-item relevance linking format
TagLinksDF = pd.DataFrame


def validate_tag_links_df(
    df: TagLinksDF,
) -> None:
    required = {
        "id_item",  # int, ItemID
        "id_tag",  # int, TagID
        "relevant",  # boolean
        "score_relevant",  # float or None
        "score_known",  # float or None
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def template_tag_links_df() -> TagLinksDF:
    return pd.DataFrame(
        {
            "id_item": [],
            "id_tag": [],
            "relevant": [],
            "score_relevant": [],
            "score_known": [],
        }
    )


# User info and rankable collection of item IDs format


@dataclass
class SampleInfo:
    """
    Data class containing fields for a user-history-rankingpool sampling, with dict converter
    Can construct from a dict by calling SampleInfo(**d)
    """

    user_id: UserID  # user ID
    history_ids: list[ItemID]  # IDs that create profile
    history_scores: list[float]  # scores of items in profile
    willrank_ids: list[ItemID]  # all IDs to rank
    # IDs within the ranking pool that the user actually watched next
    upcoming_ids: list[ItemID]
    # IDs within the ranking pool that were targeted for steerability eval
    targeted_ids: list[ItemID]

    def __str__(self) -> str:
        return pprint.pformat(
            asdict(self),
            indent=2,
        )

    def __getitem__(self, key) -> Any:
        return getattr(self, key)


# User profile format
TextProfile = str


def template_text_profile() -> TextProfile:
    return ""


# Generated ranking format
RankingDF = pd.DataFrame


def validate_ranking_df(
    df: RankingDF,
) -> None:
    required = {
        "id_item",  # int, ItemID
        "score",  # float
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def template_ranking_df() -> RankingDF:
    return pd.DataFrame(
        {
            "id_item": [],
            "score": [],
        }
    )


# Evaluation commentary format
AddtNotesDict = dict[str, list[Any]]


def template_addt_notes_dict() -> AddtNotesDict:
    return {}
