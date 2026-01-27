import os
import pickle
from typing import Callable, Optional, Tuple

from tqdm import tqdm

from custom_types import AddtNotesDict, TagID, TagInfoDF, TextProfile
from misc.item_template import get_tag_names


def naive_taglist(
    profiles_original: list[TextProfile],
    steered_tags: list[TagID],
    steer_action: str,
    df_taginfo: TagInfoDF,
    use_partial_cache: bool = False,
    partial_cache_incr: int = 10,
    partial_cache_fname: Optional[str] = None,
) -> Tuple[
    list[TextProfile],
    AddtNotesDict,
]:
    """
    Steers a "tag list" profile format by adding or removing tag names as appropriate

    :param profiles_original: Original profiles. Function assumes that these are "tag list" type profiles.
    :type profiles_original: list[TextProfile]
    :param steered_tags: Tag IDs to steer
    :type steered_tags: list[TagID]
    :param steer_action: Type of steering. Must be "increase" or "decrease".
    :type steer_action: str
    :param df_taginfo: Tag metadata
    :type df_taginfo: TagInfoDF
    :param use_partial_cache: True iff partial progress should be loaded. (Saving is determined by partial_cache_fname)
    :type use_partial_cache: bool
    :param partial_cache_incr: Update partial progress after N steerings
    :type partial_cache_incr: int
    :param partial_cache_fname: Filename to use to load/save partial progress. Required if loading progress. If non-None, fn will save partial progress.
    :type partial_cache_fname: Optional[str]
    :return: List of steered profiles, and dict containing addt notes per entry
    :rtype: Tuple[list[TextProfile], AddtNotesDict]
    """
    assert partial_cache_incr > 0

    # get target tag names (this does not need to be re-computed)
    steered_tag_names = get_tag_names(
        tag_ids=steered_tags,
        df_taginfo=df_taginfo,
    )

    # Create the steering results
    steered = []
    # Attempt loading from partial cache
    if use_partial_cache:
        assert partial_cache_fname is not None
        print(f"using partial cache: {partial_cache_fname}")
        if os.path.exists(partial_cache_fname):
            print("loading from cache")
            with open(partial_cache_fname, "rb") as f:
                (steered,) = pickle.load(f)
            print(f"already existing: {len(steered)} steered-profiles")
        else:
            print("cache not yet existing; will create")

    # now create the rest of the steered-profiles as needed
    did_update = 0
    for p in tqdm(profiles_original[len(steered) :]):
        if (
            did_update > 0
            and did_update % partial_cache_incr == 0
            and (partial_cache_fname is not None)
        ):
            print(f"updating cache at increment {did_update}")
            with open(partial_cache_fname, "wb") as f:
                pickle.dump((steered,), f)

        # break down original profile
        tag_names = p.split("\n")
        # edit the tag name collection as appropriate
        match steer_action:
            case "increase":
                tag_names = list(set(tag_names) | set(steered_tag_names))
            case "decrease":
                tag_names = list(set(tag_names) - set(steered_tag_names))
            case _:
                raise NotImplementedError
        steered.append("\n".join(tag_names))
        did_update += 1

    steered_notes = {}
    assert len(steered) == len(profiles_original)
    for notes in steered_notes.values():
        assert len(notes) == len(profiles_original)
    return (steered, steered_notes)


def default_template_append(
    tag_names: list[str],
    steer_action: str,
) -> str:
    assert len(tag_names) > 0
    # create templated sentence
    result = ""
    match steer_action:
        case "increase":
            result = "The user wants to see *more* movies that satisfy: "
        case "decrease":
            result = "The user wants to see *fewer* movies that satisfy: "
        case _:
            raise NotImplementedError
    if len(tag_names) > 1:
        result += "["
        result += ", ".join([f'"{e}"' for e in tag_names])
        result += "]"
    else:
        result += f"{tag_names[0]}"
    result += "."
    return result


def template_append_alt1(
    tag_names: list[str],
    steer_action: str,
) -> str:
    # create templated sentence
    result = ""
    match steer_action:
        case "increase":
            result = "The user wants to see more movies that satisfy: "
        case "decrease":
            result = "The user does not want to see any movies that satisfy: "
        case _:
            raise NotImplementedError
    if len(tag_names) > 1:
        result += "["
        result += ", ".join([f'"{e}"' for e in tag_names])
        result += "]"
    else:
        result += f"{tag_names[0]}"
    result += "."
    return result


def template_append_alt2(
    tag_names: list[str],
    steer_action: str,
) -> str:
    # create templated sentence
    result = ""
    match steer_action:
        case "increase":
            result = "Please show the user movies that satisfy: "
        case "decrease":
            result = "Do not show the user any movies that satisfy: "
        case _:
            raise NotImplementedError
    if len(tag_names) > 1:
        result += "["
        result += ", ".join([f'"{e}"' for e in tag_names])
        result += "]"
    else:
        result += f"{tag_names[0]}"
    result += "."
    return result


def template_append_alt3_genre(
    tag_names: list[str],
    steer_action: str,
) -> str:
    # create templated sentence
    result = ""
    match steer_action:
        case "increase":
            result = "The user wants to see more "
        case "decrease":
            result = "The user does not want to see any "
        case _:
            raise NotImplementedError
    if len(tag_names) > 1:
        result += "["
        result += ", ".join([f'"{e}"' for e in tag_names])
        result += "]"
    else:
        result += f"{tag_names[0]}"
    result += " movies."
    return result


def template_append_alt3_ddd(
    tag_names: list[str],
    steer_action: str,
) -> str:
    # create templated sentence
    result = ""
    match steer_action:
        case "increase":
            result = "The user wants to see more movies where "
        case "decrease":
            result = "The user does not want to see any movies where "
        case _:
            raise NotImplementedError
    if len(tag_names) > 1:
        result += "["
        result += ", ".join([f'"{e}"' for e in tag_names])
        result += "]"
    else:
        result += f"{tag_names[0]}"
    result += "."
    return result


def template_append_alt4(
    tag_names: list[str],
    steer_action: str,
) -> str:
    # create templated sentence
    result = ""
    match steer_action:
        case "increase":
            result = "The user *wants* to see movies that satisfy: "
        case "decrease":
            result = "The user *does not want* to see movies that satisfy: "
        case _:
            raise NotImplementedError
    if len(tag_names) > 1:
        result += "["
        result += ", ".join([f'"{e}"' for e in tag_names])
        result += "]"
    else:
        result += f"{tag_names[0]}"
    result += "."
    return result


def template_append_alt5_genre(
    tag_names: list[str],
    steer_action: str,
) -> str:
    # create templated sentence
    result = ""
    match steer_action:
        case "increase":
            result = "The user *wants* to see "
        case "decrease":
            result = "The user *does not want* to see "
        case _:
            raise NotImplementedError
    if len(tag_names) > 1:
        result += "["
        result += ", ".join([f'"{e}"' for e in tag_names])
        result += "]"
    else:
        result += f"{tag_names[0]}"
    result += " movies."
    return result


def template_append_alt5_ddd(
    tag_names: list[str],
    steer_action: str,
) -> str:
    # create templated sentence
    result = ""
    match steer_action:
        case "increase":
            result = "The user *wants* to see movies where "
        case "decrease":
            result = "The user *does not want* to see movies where "
        case _:
            raise NotImplementedError
    if len(tag_names) > 1:
        result += "["
        result += ", ".join([f'"{e}"' for e in tag_names])
        result += "]"
    else:
        result += f"{tag_names[0]}"
    result += "."
    return result


def append_template_althybrid(
    tag_names: list[str],
    steer_action: str,
) -> str:
    # create templated sentence
    result = ""
    match steer_action:
        case "increase":
            # "Please show the user movies that satisfy: Mystery."
            result = "Please show the user movies that satisfy: "
        case "decrease":
            # "The user *does not want* to see Action."
            result = "The user *does not want* to see "
        case _:
            raise NotImplementedError
    if len(tag_names) > 1:
        result += "["
        result += ", ".join([f'"{e}"' for e in tag_names])
        result += "]"
    else:
        result += f"{tag_names[0]}"
    result += "."
    return result


def append_template_alt_emph_weak_genre(
    tag_names: list[str],
    steer_action: str,
) -> str:
    # create templated sentence
    result = ""
    match steer_action:
        case "increase":
            result = "Please show the user *more* "
        case "decrease":
            result = "Please show the user *less* "
        case _:
            raise NotImplementedError
    if len(tag_names) > 1:
        result += "["
        result += ", ".join([f'"{e}"' for e in tag_names])
        result += "]"
    else:
        result += f"{tag_names[0]}"
    result += " movies."
    return result


def append_template_alt_emph_strong_genre(
    tag_names: list[str],
    steer_action: str,
) -> str:
    # create templated sentence
    result = ""
    match steer_action:
        case "increase":
            result = "Please show the user *only* "
        case "decrease":
            result = "Please show the user *no* "
        case _:
            raise NotImplementedError
    if len(tag_names) > 1:
        result += "["
        result += ", ".join([f'"{e}"' for e in tag_names])
        result += "]"
    else:
        result += f"{tag_names[0]}"
    result += " movies."
    return result


def append_template_alt_emph_weak_ddd(
    tag_names: list[str],
    steer_action: str,
) -> str:
    # create templated sentence
    result = ""
    match steer_action:
        case "increase":
            result = "The user wants to see *more* movies where "
        case "decrease":
            result = "The user wants to see *less* movies where "
        case _:
            raise NotImplementedError
    if len(tag_names) > 1:
        result += "["
        result += ", ".join([f'"{e}"' for e in tag_names])
        result += "]"
    else:
        result += f"{tag_names[0]}"
    result += "."
    return result


def append_template_alt_emph_strong_ddd(
    tag_names: list[str],
    steer_action: str,
) -> str:
    # create templated sentence
    result = ""
    match steer_action:
        case "increase":
            result = "The user wants to see *only* movies where "
        case "decrease":
            result = "The user wants to see *no* movies where "
        case _:
            raise NotImplementedError
    if len(tag_names) > 1:
        result += "["
        result += ", ".join([f'"{e}"' for e in tag_names])
        result += "]"
    else:
        result += f"{tag_names[0]}"
    result += "."
    return result


def append_template(
    profiles_original: list[TextProfile],
    steered_tags: list[TagID],
    steer_action: str,
    df_taginfo: TagInfoDF,
    template: Callable[
        [
            list[str],
            str,
        ],
        str,
    ] = default_template_append,
) -> Tuple[
    list[TextProfile],
    AddtNotesDict,
]:
    """
    Steers a profile by adding a templated sentence at the end.

    :param profiles_original: Original profiles
    :type profiles_original: list[TextProfile]
    :param steered_tags: Tag IDs to steer
    :type steered_tags: list[TagID]
    :param steer_action: Type of steering. Must be "increase" or "decrease".
    :type steer_action: str
    :param df_taginfo: Tag metadata
    :type df_taginfo: TagInfoDF
    :param template: Function to create template sentence based on a list of tag names
    :type template: Callable[[list[str], str], str]
    :return: List of steered profiles, and dict containing addt notes per entry
    :rtype: Tuple[list[TextProfile], AddtNotesDict]
    """
    # get target tag names
    steered_tag_names = get_tag_names(
        tag_ids=steered_tags,
        df_taginfo=df_taginfo,
    )
    # Create the steering results
    steered = [
        " ".join(
            [
                p,
                template(
                    steered_tag_names,
                    steer_action,
                ),
            ]
        )
        for p in profiles_original
    ]
    steered_notes = {}
    return (steered, steered_notes)


def default_template_replace(
    tag_names: list[str],
    steer_action: str,
) -> str:
    assert len(tag_names) > 0
    # create templated sentence
    result = ""
    match steer_action:
        case "increase":
            result = "The user *wants* movies that satisfy: "
        case "decrease":
            result = "The user *does not want* movies that satisfy: "
        case _:
            raise NotImplementedError
    if len(tag_names) > 1:
        result += "["
        result += ", ".join([f'"{e}"' for e in tag_names])
        result += "]."
    else:
        result += f"{tag_names[0]}."
    return result


def replace(
    profiles_original: list[TextProfile],
    steered_tags: list[TagID],
    steer_action: str,
    df_taginfo: TagInfoDF,
    template: Callable[
        [
            list[str],
            str,
        ],
        str,
    ] = default_template_replace,
) -> Tuple[
    list[TextProfile],
    AddtNotesDict,
]:
    """
    Steers a profile by fully replacing it with a templated sentence.

    :param profiles_original: Original profiles
    :type profiles_original: list[TextProfile]
    :param steered_tags: Tag IDs to steer
    :type steered_tags: list[TagID]
    :param steer_action: Type of steering. Must be "increase" or "decrease".
    :type steer_action: str
    :param df_taginfo: Tag metadata
    :type df_taginfo: TagInfoDF
    :param template: Function to create template sentence based on a list of tag names
    :type template: Callable[[list[str], str], str]
    :return: List of steered profiles, and dict containing addt notes per entry
    :rtype: Tuple[list[TextProfile], AddtNotesDict]
    """
    # get target tag names
    steered_tag_names = get_tag_names(
        tag_ids=steered_tags,
        df_taginfo=df_taginfo,
    )
    # Create the steering results
    steered = [
        " ".join(
            [
                p,
                template(
                    steered_tag_names,
                    steer_action,
                ),
            ]
        )
        for p in profiles_original
    ]
    steered_notes = {}
    return (steered, steered_notes)
