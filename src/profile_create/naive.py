import os
import pickle
from typing import Optional, Tuple

from tqdm import tqdm

from custom_types import (
    AddtNotesDict,
    ItemID,
    ItemInfoDict,
    RatingDataDF,
    TextProfile,
    UserID,
)
from custom_types import SampleInfo
from custom_types import TagID, TagInfoDF, TagLinksDF
from misc.item_template import get_tag_names, templatify_item_id


def naive_empty(
    per_user_samples: list[SampleInfo],
    profile_text: str = "",
) -> Tuple[
    list[TextProfile],
    AddtNotesDict,
]:
    """
    Creates empty profiles (or use the passed content string)

    :param per_user_samples: Experiment samples
    :type per_user_samples: list[SampleInfo]
    :param profile_text: Content string to define all profiles
    :type profile_text: str
    :return: List of profiles, and dict containing addt notes per entry
    :rtype: Tuple[list[TextProfile], AddtNotesDict]
    """
    profiles = [profile_text for e in per_user_samples]
    profile_notes = {}
    return (profiles, profile_notes)


def naive_taglist(
    per_user_samples: list[SampleInfo],
    df_taginfo: TagInfoDF,
    df_taglinks: TagLinksDF,
    use_partial_cache: bool = False,
    partial_cache_incr: int = 10,
    partial_cache_fname: Optional[str] = None,
) -> Tuple[
    list[TextProfile],
    AddtNotesDict,
]:
    """
    Creates profiles consisting of all tags related to items in history,
    translated to words and broken up by lines

    :param per_user_samples: Experiment samples
    :type per_user_samples: list[SampleInfo]
    :param df_taginfo: Info about tags
    :type df_taginfo: TagInfoDF
    :param df_taglinks: Info about relationship between tags and items
    :type df_taglinks: TagLinksDF
    :param use_partial_cache: True iff partial progress should be loaded. (Saving is determined by partial_cache_fname)
    :type use_partial_cache: bool
    :param partial_cache_incr: Update partial progress after N profiles
    :type partial_cache_incr: int
    :param partial_cache_fname: Filename to use to load/save partial progress. Required if loading progress. If non-None, fn will save partial progress.
    :type partial_cache_fname: Optional[str]
    :return: List of profiles, and dict containing addt notes per entry
    :rtype: Tuple[list[TextProfile], AddtNotesDict]
    """
    assert partial_cache_incr > 0

    # Create the profiles results
    profiles = []
    # Attempt loading from partial cache
    if use_partial_cache:
        assert partial_cache_fname is not None
        print(f"using partial cache: {partial_cache_fname}")
        if os.path.exists(partial_cache_fname):
            print("loading from cache")
            with open(partial_cache_fname, "rb") as f:
                (profiles,) = pickle.load(f)
            print(f"already existing: {len(profiles)} profiles")
        else:
            print("cache not yet existing; will create")

    # now create the rest of the profiles as needed
    did_update = 0
    for u in tqdm(per_user_samples[len(profiles) :]):
        if (
            did_update > 0
            and did_update % partial_cache_incr == 0
            and (partial_cache_fname is not None)
        ):
            print(f"updating cache at increment {did_update}")
            with open(partial_cache_fname, "wb") as f:
                pickle.dump((profiles,), f)

        # Get list of all tags related to all items from user history
        relevant_per_item = [
            df_taglinks[
                (df_taglinks["relevant"] == True) & (df_taglinks["id_item"] == item_id)
            ]["id_tag"].tolist()
            for item_id in u.history_ids
        ]
        relevant_tags = list(set([e for sl in relevant_per_item for e in sl]))
        # Profileify them
        profiles.append(
            "\n".join(
                get_tag_names(
                    tag_ids=relevant_tags,
                    df_taginfo=df_taginfo,
                )
            )
        )
        did_update += 1

    profile_notes = {}
    assert len(profiles) == len(per_user_samples)
    for notes in profile_notes.values():
        assert len(notes) == len(per_user_samples)
    return (profiles, profile_notes)


def list_movies(
    per_user_samples: list[SampleInfo],
    profile_data: str,
    df_ratings: RatingDataDF,
    item_info: ItemInfoDict,
    df_taginfo: TagInfoDF,
    df_taglinks: TagLinksDF,
    use_partial_cache: bool = False,
    partial_cache_incr: int = 10,
    partial_cache_fname: Optional[str] = None,
) -> Tuple[
    list[TextProfile],
    AddtNotesDict,
]:
    """
    Creates profiles consisting of metadata related to items in history,
    converted to single-linebroken sections, separated by empty lines

    :param per_user_samples: Experiment samples
    :type per_user_samples: list[SampleInfo]
    :param profile_data: Type of metadata to include about items
    :type profile_data: str
    :param df_ratings: Rating information per user/item
    :type df_ratings: RatingDataDF
    :param item_info: Metadata per item
    :type item_info: ItemInfoDict
    :param df_taginfo: Info about tags
    :type df_taginfo: TagInfoDF
    :param df_taglinks: Info about relationship between tags and items
    :type df_taglinks: TagLinksDF
    :param use_partial_cache: True iff partial progress should be loaded. (Saving is determined by partial_cache_fname)
    :type use_partial_cache: bool
    :param partial_cache_incr: Update partial progress after N profiles
    :type partial_cache_incr: int
    :param partial_cache_fname: Filename to use to load/save partial progress. Required if loading progress. If non-None, fn will save partial progress.
    :type partial_cache_fname: Optional[str]
    :return: List of profiles, and dict containing addt notes per entry
    :rtype: Tuple[list[TextProfile], AddtNotesDict]
    """
    assert partial_cache_incr > 0

    # Create the profiles results
    profiles = []
    # Attempt loading from partial cache
    if use_partial_cache:
        assert partial_cache_fname is not None
        print(f"using partial cache: {partial_cache_fname}")
        if os.path.exists(partial_cache_fname):
            print("loading from cache")
            with open(partial_cache_fname, "rb") as f:
                (profiles,) = pickle.load(f)
            print(f"already existing: {len(profiles)} profiles")
        else:
            print("cache not yet existing; will create")

    # now create the rest of the profiles as needed
    did_update = 0
    for u in tqdm(per_user_samples[len(profiles) :]):
        if (
            did_update > 0
            and did_update % partial_cache_incr == 0
            and (partial_cache_fname is not None)
        ):
            print(f"updating cache at increment {did_update}")
            with open(partial_cache_fname, "wb") as f:
                pickle.dump((profiles,), f)

        profile_text = "".join(
            [
                "Movies watched:",
            ]
            + [
                templatify_item_id(
                    metadata=profile_data,
                    user_id=u.user_id,
                    item_id=e,
                    df_ratings=df_ratings,
                    item_info=item_info,
                    df_taginfo=df_taginfo,
                    df_taglinks=df_taglinks,
                    use_score=True,
                )
                for e in u.history_ids
            ]
        )
        profiles.append(profile_text)
        did_update += 1

    profile_notes = {}
    assert len(profiles) == len(per_user_samples)
    for notes in profile_notes.values():
        assert len(notes) == len(per_user_samples)
    return (profiles, profile_notes)
