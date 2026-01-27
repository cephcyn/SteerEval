import os
import pickle
from typing import Callable, Optional, Tuple

from tqdm import tqdm

from custom_types import (
    AddtNotesDict,
    ItemInfoDict,
    RatingDataDF,
    SampleInfo,
    TagInfoDF,
    TagLinksDF,
    TextProfile,
)
from misc.item_template import templatify_item_id
from misc.llm_tools import model_instruct_generate

DEFAULT_SYSTEM_PROMPT = (
    "Write a response that appropriately completes the following request."
    " Follow the instructions exactly."
)


def default_prompt_profile(
    instructions: list[str],
    item_blurbs: list[str],
) -> str:
    return "\n\n".join(instructions + item_blurbs)


def llm_profile(
    per_user_samples: list[SampleInfo],
    main_prompt: str,
    profile_data: str,
    df_ratings: RatingDataDF,
    item_info: ItemInfoDict,
    df_taginfo: TagInfoDF,
    df_taglinks: TagLinksDF,
    tokenizer,
    model,
    template: Callable[
        [
            list[str],
            list[str],
        ],
        str,
    ] = default_prompt_profile,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    seed: Optional[int] = None,
    use_partial_cache: bool = False,
    partial_cache_incr: int = 10,
    partial_cache_fname: Optional[str] = None,
) -> Tuple[
    list[TextProfile],
    AddtNotesDict,
]:
    """
    Creates profiles by prompting a LLM with instruction and metadata related to items in history.

    :param per_user_samples: Experiment samples
    :type per_user_samples: list[SampleInfo]
    :param main_prompt: General prompt to use to instruct LLM for profile creation
    :type main_prompt: str
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
    :param tokenizer: Model tokenizer object. Must be able to call apply_chat_template
    :param model: Some AutoModelForCausalLM-loadable object
    :param system_prompt: System prompt to use
    :type system_prompt: str
    :param seed: Random seed used for LLM prompting if not None
    :type seed: Optional[int]
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
    note_prompt_system = []
    note_prompt_main = []
    # Attempt loading from partial cache
    if use_partial_cache:
        assert partial_cache_fname is not None
        print(f"using partial cache: {partial_cache_fname}")
        if os.path.exists(partial_cache_fname):
            print("loading from cache")
            with open(partial_cache_fname, "rb") as f:
                (
                    profiles,
                    note_prompt_system,
                    note_prompt_main,
                ) = pickle.load(f)
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
                pickle.dump(
                    (
                        profiles,
                        note_prompt_system,
                        note_prompt_main,
                    ),
                    f,
                )

        constructed_main_prompt = template(
            [main_prompt],
            [
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
            ],
        )
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": constructed_main_prompt,
            },
        ]
        profile_text = model_instruct_generate(
            tokenizer=tokenizer,
            model=model,
            messages=messages,
            seed=seed,
        )
        note_prompt_system.append(system_prompt)
        note_prompt_main.append(constructed_main_prompt)
        profiles.append(profile_text)
        did_update += 1

    profile_notes = {
        "prompt_system": note_prompt_system,
        "prompt_main": note_prompt_main,
    }
    assert len(profiles) == len(per_user_samples)
    for notes in profile_notes.values():
        assert len(notes) == len(per_user_samples)
    return (profiles, profile_notes)


def llm_profile_sentence(
    per_user_samples: list[SampleInfo],
    profile_data: str,
    df_ratings: RatingDataDF,
    item_info: ItemInfoDict,
    df_taginfo: TagInfoDF,
    df_taglinks: TagLinksDF,
    tokenizer,
    model,
    seed: Optional[int] = None,
    use_partial_cache: bool = False,
    partial_cache_incr: int = 10,
    partial_cache_fname: Optional[str] = None,
) -> Tuple[
    list[TextProfile],
    AddtNotesDict,
]:
    """
    Prompt LLM to generate single-sentence profile based on item history metadata.

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
    :param tokenizer: Model tokenizer object. Must be able to call apply_chat_template
    :param model: Some AutoModelForCausalLM-loadable object
    :param seed: Random seed used for LLM prompting if not None
    :type seed: Optional[int]
    :param use_partial_cache: True iff partial progress should be loaded. (Saving is determined by partial_cache_fname)
    :type use_partial_cache: bool
    :param partial_cache_incr: Update partial progress after N profiles
    :type partial_cache_incr: int
    :param partial_cache_fname: Filename to use to load/save partial progress. Required if loading progress. If non-None, fn will save partial progress.
    :type partial_cache_fname: Optional[str]
    :return: List of profiles, and dict containing addt notes per entry
    :rtype: Tuple[list[TextProfile], AddtNotesDict]
    """
    return llm_profile(
        per_user_samples=per_user_samples,
        main_prompt=(
            "Given the user's previously watched and rated movies,"
            " write exactly ONE concise English sentence (20-35 words)"
            " summarizing their taste. Be definitive (no hedging)."
            " Do NOT mention tags, genres, metadata, reviews, or any lack thereof."
            " Infer tone, pacing, themes, and style."
            "\n\n"
            "User watch history:"
        ),
        profile_data=profile_data,
        df_ratings=df_ratings,
        item_info=item_info,
        df_taginfo=df_taginfo,
        df_taglinks=df_taglinks,
        tokenizer=tokenizer,
        model=model,
        use_partial_cache=use_partial_cache,
        partial_cache_incr=partial_cache_incr,
        partial_cache_fname=partial_cache_fname,
    )


def llm_profile_paragraph(
    per_user_samples: list[SampleInfo],
    profile_data: str,
    df_ratings: RatingDataDF,
    item_info: ItemInfoDict,
    df_taginfo: TagInfoDF,
    df_taglinks: TagLinksDF,
    tokenizer,
    model,
    seed: Optional[int] = None,
    use_partial_cache: bool = False,
    partial_cache_incr: int = 10,
    partial_cache_fname: Optional[str] = None,
) -> Tuple[
    list[TextProfile],
    AddtNotesDict,
]:
    """
    Prompt LLM to generate a paragraph profile based on item history metadata.

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
    :param tokenizer: Model tokenizer object. Must be able to call apply_chat_template
    :param model: Some AutoModelForCausalLM-loadable object
    :param seed: Random seed used for LLM prompting if not None
    :type seed: Optional[int]
    :param use_partial_cache: True iff partial progress should be loaded. (Saving is determined by partial_cache_fname)
    :type use_partial_cache: bool
    :param partial_cache_incr: Update partial progress after N profiles
    :type partial_cache_incr: int
    :param partial_cache_fname: Filename to use to load/save partial progress. Required if loading progress. If non-None, fn will save partial progress.
    :type partial_cache_fname: Optional[str]
    :return: List of profiles, and dict containing addt notes per entry
    :rtype: Tuple[list[TextProfile], AddtNotesDict]
    """
    return llm_profile(
        per_user_samples=per_user_samples,
        main_prompt=(
            "Given the user's previously watched and rated movies,"
            " write in a single paragraph (5-6 sentences)"
            " summarizing their taste. Be definitive (no hedging)."
            " Do NOT mention tags, genres, metadata, reviews, or any lack thereof."
            " Infer tone, pacing, themes, and style."
            "\n\n"
            "User watch history:"
        ),
        profile_data=profile_data,
        df_ratings=df_ratings,
        item_info=item_info,
        df_taginfo=df_taginfo,
        df_taglinks=df_taglinks,
        tokenizer=tokenizer,
        model=model,
        use_partial_cache=use_partial_cache,
        partial_cache_incr=partial_cache_incr,
        partial_cache_fname=partial_cache_fname,
    )
