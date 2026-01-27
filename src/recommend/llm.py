import os
import pickle
import random
import re
from typing import Callable, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from custom_types import (
    AddtNotesDict,
    ItemID,
    ItemInfoDict,
    RankingDF,
    RatingDataDF,
    SampleInfo,
    TagID,
    TagInfoDF,
    TagLinksDF,
    TextProfile,
)
from misc.item_template import templatify_item_id, templatify_item_id_oneline
from misc.llm_tools import model_instruct_generate

DEFAULT_SYSTEM_PROMPT = (
    "Write a response that appropriately completes the following request."
    " Follow the instructions exactly."
)


def default_prompt_ordering(
    user_profile: str,
    item_ids: list[ItemID],
    item_blurbs: list[str],
) -> str:
    # create templated LLM prompt to do item ordering ranking task
    result = ""
    result += "USER PROFILE:" "\n" f"{user_profile}" "\n\n"
    result += f"MOVIE CANDIDATES: (each line is one movie, numbered):"
    result += "\n"
    for i in range(len(item_ids)):
        result += f"{i+1}. {item_blurbs[i]}"
        result += "\n"
    result += "\n"
    result += (
        "Task: Sort all of these movies from BEST to WORST match for the user."
        "\n"
        "Guidelines:"
        "\n"
        "- Output a list of movies, where each line is formatted like: `1. Movie Name (Movie ID) - Short Explanation`"
        "\n"
        "- You may reason about the ranking before writing a final result."
        # "\n"
        # "- Every movie ID must be included in the ranking."
        # "\n"
        # "- Do not include any duplicates in the ranking."
        # "\n"
        # "- The list must be in descending order of match quality."
    )
    return result


def default_postprocess_ordering(
    text: str,
    sample_info: SampleInfo,
) -> list[ItemID]:
    # assume the relevant lines take format "1. Movie Name (Movie ID) - Other Text"
    text_lines = text.split()
    output_pos = []
    for line in text_lines:
        match = re.search(r"\d+\.\s*.+\((d+)\)\s*-\s*.*", line)
        if match is not None:
            match_str = match.group(1)  # first relevant grouping, the ID
            output_pos.append(int(match_str.strip()))
    # convert ordered list of item positions into item IDs
    output_ids = [
        sample_info.willrank_ids[i]
        for i in output_pos
        if ((i >= 0) and (i < len(sample_info.willrank_ids)))
    ]
    return output_ids


def postprocess_ordering_intlist(
    text: str,
    sample_info: SampleInfo,
) -> list[ItemID]:
    # earlier in the list is better match, later is worse
    match = re.search(r"\[[\d\s,\'\"]+\]?", text)
    if match is None:
        return []
    else:
        match_str = match.group()  # full regex hit
        match_str = match_str.strip("[]")
        match_ids = [int(e.strip()) for e in match_str.split(",") if len(e.strip()) > 0]
        return match_ids


def llm_ordering(
    per_user_samples: list[SampleInfo],
    profiles: list[TextProfile],
    item_data: str,
    df_ratings: RatingDataDF,
    item_info: ItemInfoDict,
    df_taginfo: TagInfoDF,
    df_taglinks: TagLinksDF,
    tokenizer,
    model,
    targeted_tag_ids: Optional[list[TagID]] = None,
    template: Callable[
        [
            str,
            list[ItemID],
            list[str],
        ],
        str,
    ] = default_prompt_ordering,
    postprocess: Callable[
        [
            str,
            SampleInfo,
        ],
        list[ItemID],
    ] = default_postprocess_ordering,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    rng: Optional[random.Random] = None,
    seed: Optional[int] = None,
    use_partial_cache: bool = False,
    partial_cache_incr: int = 10,
    partial_cache_fname: Optional[str] = None,
) -> Tuple[
    list[RankingDF],
    AddtNotesDict,
]:
    """
    Rank items by prompting instruct-tuned LLM to create an ordered ranking given all of the items at once.
    Items missing from the ranking are randomly appended to the end.

    :param per_user_samples: Experiment samples
    :type per_user_samples: list[SampleInfo]
    :param profiles: User profiles
    :type profiles: list[TextProfile]
    :param item_data: Type of metadata to include about items
    :type item_data: str
    :param df_ratings: Rating information per user/item
    :type df_ratings: RatingDataDF
    :param item_info: Metadata per item
    :type item_info: ItemInfoDict
    :param df_taginfo: Tag metadata
    :type df_taginfo: TagInfoDF
    :param df_taglinks: Tag links data
    :type df_taglinks: TagLinksDF
    :param tokenizer: Model tokenizer object. Must be able to call apply_chat_template
    :param model: Some AutoModelForCausalLM-loadable object
    :param targeted_tag_ids: Targeted tag ID, if item_data="target_tags"
    :type targeted_tag_ids: Optional[list[TagID]]
    :param template: Function to create template LLM prompt based on a profile text, item IDs, and item blurbs.
    :type template: Callable[[str, list[ItemID], list[str]], str,]
    :param postprocess: Function to postprocess LLM output into a list of item IDs (from best to worst)
    :type postprocess: Callable[[str], list[ItemID],]
    :param system_prompt: System prompt to use
    :type system_prompt: str
    :param rng: Random object to use. Takes priority over seed.
    :type rng: Optional[random.Random]
    :param seed: Seed for random object and LLM prompting to use.
    :type seed: Optional[int]
    :param use_partial_cache: True iff partial progress should be loaded. (Saving is determined by partial_cache_fname)
    :type use_partial_cache: bool
    :param partial_cache_incr: Update partial progress after N rankings
    :type partial_cache_incr: int
    :param partial_cache_fname: Filename to use to load/save partial progress. Required if loading progress. If non-None, fn will save partial progress.
    :type partial_cache_fname: Optional[str]
    :return: List of rankings, and dict containing extra notes per entry
    :rtype: Tuple[list[RankingDF], AddtNotesDict]
    """
    assert len(per_user_samples) == len(profiles)
    assert partial_cache_incr > 0

    # create randomizer
    rng_used = random.Random(seed)
    if rng is not None:
        rng_used = rng

    # Create the rankings results
    rankings: list[RankingDF] = []
    note_prompt_system: list[str] = []
    note_prompt_main: list[str] = []
    note_output_raw: list[str] = []
    note_output_extracted: list[list[ItemID]] = []
    # Attempt loading from partial cache
    if use_partial_cache:
        assert partial_cache_fname is not None
        print(f"using partial cache: {partial_cache_fname}")
        if os.path.exists(partial_cache_fname):
            print("loading from cache")
            with open(partial_cache_fname, "rb") as f:
                (
                    rankings,
                    note_prompt_system,
                    note_prompt_main,
                    note_output_raw,
                    note_output_extracted,
                ) = pickle.load(f)
            print(f"already existing: {len(rankings)} rankings")
        else:
            print("cache not yet existing; will create")

    # now create the rest of the rankings as needed
    did_update = 0
    for u, p in tqdm(
        list(
            zip(
                per_user_samples[len(rankings) :],
                profiles[len(rankings) :],
            )
        )
    ):
        if (
            did_update > 0
            and did_update % partial_cache_incr == 0
            and (partial_cache_fname is not None)
        ):
            print(f"updating cache at increment {did_update}")
            with open(partial_cache_fname, "wb") as f:
                pickle.dump(
                    (
                        rankings,
                        note_prompt_system,
                        note_prompt_main,
                        note_output_raw,
                        note_output_extracted,
                    ),
                    f,
                )

        # construct main prompt for ranking
        constructed_main_prompt = template(
            p,
            u.willrank_ids,
            [
                templatify_item_id_oneline(
                    metadata=item_data,
                    user_id=u.user_id,
                    item_id=e,
                    df_ratings=df_ratings,
                    item_info=item_info,
                    df_taginfo=df_taginfo,
                    df_taglinks=df_taglinks,
                    use_score=False,
                    targeted_tag_ids=targeted_tag_ids,
                )
                for e in u.willrank_ids
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
        # generate output
        model_output = model_instruct_generate(
            tokenizer=tokenizer,
            model=model,
            messages=messages,
            seed=seed,
        )
        # postprocess output to get ordered list
        result_list = postprocess(model_output, u)
        # remove invalid and redundant IDs
        result_list_valid: list[ItemID] = []
        was_seen = set()
        for i in result_list:
            if i not in was_seen:
                was_seen.add(i)
                if i in u.willrank_ids:
                    result_list_valid.append(i)
        # handle missing IDs
        missing_ids = list(set(u.willrank_ids) - set(result_list_valid))
        # create synthetic scoring for the ranking object
        synthetic_scores = list(reversed(range(len(result_list_valid)))) + [
            (-1 * rng_used.random()) for _ in missing_ids
        ]
        # append any excluded IDs at the end
        u_ranking: RankingDF = pd.DataFrame(
            {
                "id_item": result_list_valid + missing_ids,
                "score": synthetic_scores,
            }
        )
        u_ranking = u_ranking.sort_values(
            by="score",
            ascending=False,
            na_position="last",
        )
        note_prompt_system.append(system_prompt)
        note_prompt_main.append(constructed_main_prompt)
        note_output_raw.append(model_output)
        note_output_extracted.append(result_list)
        rankings.append(u_ranking)
        did_update += 1

    rankings_notes: AddtNotesDict = {
        "prompt_system": note_prompt_system,
        "prompt_main": note_prompt_main,
        "output_raw": note_output_raw,
        "output_extracted": note_output_extracted,
    }
    assert len(rankings) == len(per_user_samples)
    for notes in rankings_notes.values():
        assert len(notes) == len(per_user_samples)
    return (rankings, rankings_notes)


def default_prompt_scorepred(
    user_profile: str,
    item_blurb: str,
) -> str:
    # create templated LLM prompt to do score prediction task
    result = ""
    result += (
        "Task: Predict the user's rating for a movie on a scale from 0.0 to 5.0"
        "\n"
        "Guidelines:"
        "\n"
        "- 0.0: Completely irrelevant."
        "\n"
        "- 5.0: Perfect match for ALL user interests."
        "\n"
        "- Use decimal precision as necessary (e.g. 1.4, 3.7, 4.2) to capture partial matches."
        "\n"
        "- Be strict. Do not give high scores for partial matches."
        "\n"
        "Output format: A single float number only."
        "\n\n"
        "USER PROFILE:"
        "\n"
        f"{user_profile}"
        "\n"
        "CANDIDATE MOVIE:"
        "\n"
        f"{item_blurb}"
        "\n"
        "SCORE:"
    )
    return result


def default_postprocess_scorepred(
    text: str,
    sample_info: SampleInfo,
) -> float:
    # higher score is better match, lower is worse
    match = re.search(r"[0-9]+\.?[0-9]*", text)
    if match is None:
        return -1
    else:
        return float(match.group())


def llm_scorepred(
    per_user_samples: list[SampleInfo],
    profiles: list[TextProfile],
    item_data: str,
    df_ratings: RatingDataDF,
    item_info: ItemInfoDict,
    df_taginfo: TagInfoDF,
    df_taglinks: TagLinksDF,
    tokenizer,
    model,
    targeted_tag_ids: Optional[list[TagID]] = None,
    template: Callable[
        [
            str,
            str,
        ],
        str,
    ] = default_prompt_scorepred,
    postprocess: Callable[
        [
            str,
            SampleInfo,
        ],
        float,
    ] = default_postprocess_scorepred,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    rng: Optional[random.Random] = None,
    seed: Optional[int] = None,
    tiebreak_magnitude: float = 0.0005,
    use_partial_cache: bool = False,
    partial_cache_incr: int = 10,
    partial_cache_fname: Optional[str] = None,
) -> Tuple[
    list[RankingDF],
    AddtNotesDict,
]:
    """
    Rank items by prompting instruct-tuned LLM to predict item score.
    Tiebreaks are done by adding a small amount of random noise to all item scores after extracting them.

    :param per_user_samples: Experiment samples
    :type per_user_samples: list[SampleInfo]
    :param profiles: User profiles
    :type profiles: list[TextProfile]
    :param item_data: Type of metadata to include about items
    :type item_data: str
    :param df_ratings: Rating information per user/item
    :type df_ratings: RatingDataDF
    :param item_info: Metadata per item
    :type item_info: ItemInfoDict
    :param df_taginfo: Tag metadata
    :type df_taginfo: TagInfoDF
    :param df_taglinks: Tag links data
    :type df_taglinks: TagLinksDF
    :param tokenizer: Model tokenizer object. Must be able to call apply_chat_template
    :param model: Some AutoModelForCausalLM-loadable object
    :param targeted_tag_ids: Targeted tag ID, if item_data="target_tags"
    :type targeted_tag_ids: Optional[list[TagID]]
    :param template: Function to create template LLM prompt based on a profile text and an item metadata blurb text.
    :type template: Callable[[str, str], str,]
    :param postprocess: Function to postprocess LLM output into a float score
    :type postprocess: Callable[[str], float,]
    :param system_prompt: System prompt to use
    :type system_prompt: str
    :param rng: Random object to use. Takes priority over seed.
    :type rng: Optional[random.Random]
    :param seed: Seed for random object and LLM prompting to use.
    :type seed: Optional[int]
    :param tiebreak_magnitude: Magnitude to limit tiebreaking to
    :type tiebreak_magnitude: float
    :param use_partial_cache: True iff partial progress should be loaded. (Saving is determined by partial_cache_fname)
    :type use_partial_cache: bool
    :param partial_cache_incr: Update partial progress after N rankings
    :type partial_cache_incr: int
    :param partial_cache_fname: Filename to use to load/save partial progress. Required if loading progress. If non-None, fn will save partial progress.
    :type partial_cache_fname: Optional[str]
    :return: List of rankings, and dict containing extra notes per entry
    :rtype: Tuple[list[RankingDF], AddtNotesDict]
    """
    assert len(per_user_samples) == len(profiles)
    assert partial_cache_incr > 0

    # create randomizer
    rng_used = random.Random(seed)
    if rng is not None:
        rng_used = rng

    # Create the rankings results
    rankings: list[RankingDF] = []
    note_prompt_system: list[str] = []
    note_prompts_main: list[list[str]] = []
    note_outputs_raw: list[list[str]] = []
    note_outputs_processed: list[list[float]] = []
    # Attempt loading from partial cache
    if use_partial_cache:
        assert partial_cache_fname is not None
        print(f"using partial cache: {partial_cache_fname}")
        if os.path.exists(partial_cache_fname):
            print("loading from cache")
            with open(partial_cache_fname, "rb") as f:
                (
                    rankings,
                    note_prompt_system,
                    note_prompts_main,
                    note_outputs_raw,
                    note_outputs_processed,
                ) = pickle.load(f)
            print(f"already existing: {len(rankings)} rankings")
        else:
            print("cache not yet existing; will create")

    # now create the rest of the rankings as needed
    did_update = 0
    for u, p in tqdm(
        list(
            zip(
                per_user_samples[len(rankings) :],
                profiles[len(rankings) :],
            )
        )
    ):
        if (
            did_update > 0
            and did_update % partial_cache_incr == 0
            and (partial_cache_fname is not None)
        ):
            print(f"updating cache at increment {did_update}")
            with open(partial_cache_fname, "wb") as f:
                pickle.dump(
                    (
                        rankings,
                        note_prompt_system,
                        note_prompts_main,
                        note_outputs_raw,
                        note_outputs_processed,
                    ),
                    f,
                )

        # construct main prompt per item
        constructed_main_prompts = [
            template(
                p,
                templatify_item_id(
                    metadata=item_data,
                    user_id=u.user_id,
                    item_id=e,
                    df_ratings=df_ratings,
                    item_info=item_info,
                    df_taginfo=df_taginfo,
                    df_taglinks=df_taglinks,
                    use_score=False,
                    targeted_tag_ids=targeted_tag_ids,
                ),
            )
            for e in u.willrank_ids
        ]
        all_messages = [
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": p,
                },
            ]
            for p in constructed_main_prompts
        ]
        # generate output
        all_model_outputs = [
            model_instruct_generate(
                tokenizer=tokenizer,
                model=model,
                messages=m,
                seed=seed,
            )
            for m in all_messages
        ]
        # postprocess output to extract float scores
        result_scores = [postprocess(s, u) for s in all_model_outputs]
        # tiebreaking noise
        result_scores_noisy = [
            s + (rng_used.random() * tiebreak_magnitude) for s in result_scores
        ]
        u_ranking: RankingDF = pd.DataFrame(
            {
                "id_item": u.willrank_ids,
                "score": result_scores_noisy,
            }
        )
        u_ranking = u_ranking.sort_values(
            by="score",
            ascending=False,
            na_position="last",
        )
        note_prompt_system.append(system_prompt)
        note_prompts_main.append(constructed_main_prompts)
        note_outputs_raw.append(all_model_outputs)
        note_outputs_processed.append(result_scores)
        rankings.append(u_ranking)
        did_update += 1

    rankings_notes: AddtNotesDict = {
        "prompt_system": note_prompt_system,
        "prompts_main": note_prompts_main,
        "outputs_raw": note_outputs_raw,
        "outputs_processed": note_outputs_processed,
    }
    assert len(rankings) == len(per_user_samples)
    for notes in rankings_notes.values():
        assert len(notes) == len(per_user_samples)
    return (rankings, rankings_notes)
