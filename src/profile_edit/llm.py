import os
import pickle
from typing import Callable, Optional, Tuple

from tqdm import tqdm
from custom_types import AddtNotesDict, TagID, TagInfoDF, TextProfile
from misc.item_template import get_tag_names
from misc.llm_tools import model_instruct_generate


DEFAULT_SYSTEM_PROMPT = (
    "Write a response that appropriately completes the following request."
    " Follow the instructions exactly."
)


def default_prompt_append(
    tag_names: list[str],
    steer_action: str,
) -> str:
    assert len(tag_names) > 0
    # create templated LLM prompt to get a natural-sounding append sentence
    result = ""
    result += (
        "You are helping a user update how they describe their movie preferences"
        " by rephrasing their changed preferences into a single sentence."
        " This sentence will be appended to their original description."
        "\n"
        "Guidelines:"
        "\n"
        '- Write exactly *one* sentence. The sentence must start with "The user".'
        "\n"
        '- Do not include any "REQUEST" in the response. Only give the rephrased sentence.'
        "\n"
        "- Use natural language."
        "\n\n"
        "For example:"
        "\n"
        "REQUEST: The user wants to see *fewer* movies that satisfy: Does the dog die."
        "\n"
        "The user does not want to watch movies where a dog dies."
        "\n\n"
    )
    match steer_action:
        case "increase":
            result += "REQUEST: The user wants to see *more* movies that satisfy: "
        case "decrease":
            result += "REQUEST: The user wants to see *fewer* movies that satisfy: "
        case _:
            raise NotImplementedError
    if len(tag_names) > 1:
        result += "["
        result += ", ".join([f'"{e}"' for e in tag_names])
        result += "]."
    else:
        result += f"{tag_names[0]}."
    return result


def prompt_append_alt5_genre(
    tag_names: list[str],
    steer_action: str,
) -> str:
    assert len(tag_names) > 0
    # create templated LLM prompt to get a natural-sounding append sentence
    result = ""
    result += (
        "You are helping a user update how they describe their movie preferences"
        " by rephrasing their changed preferences into a single sentence."
        " This sentence will be appended to their original description."
        "\n"
        "Guidelines:"
        "\n"
        '- Write exactly *one* sentence. The sentence must start with "The user".'
        "\n"
        '- Do not include any "REQUEST" in the response. Only give the rephrased sentence.'
        "\n"
        "- Use natural language."
        "\n\n"
        "For example:"
        "\n"
        "REQUEST: The user *does not want* to see [Romance] movies."
        "\n"
        "The user *does not want* to watch any romance movies."
        "\n\n"
    )
    match steer_action:
        case "increase":
            result += "REQUEST: The user *wants* to see "
        case "decrease":
            result += "REQUEST: The user *does not want* to see "
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


def prompt_append_alt5_ddd(
    tag_names: list[str],
    steer_action: str,
) -> str:
    assert len(tag_names) > 0
    # create templated LLM prompt to get a natural-sounding append sentence
    result = ""
    result += (
        "You are helping a user update how they describe their movie preferences"
        " by rephrasing their changed preferences into a single sentence."
        " This sentence will be appended to their original description."
        "\n"
        "Guidelines:"
        "\n"
        '- Write exactly *one* sentence. The sentence must start with "The user".'
        "\n"
        '- Do not include any "REQUEST" in the response. Only give the rephrased sentence.'
        "\n"
        "- Use natural language."
        "\n\n"
        "For example:"
        "\n"
        "REQUEST: The user *does not want* to see movies where [the dog dies]."
        "\n"
        "The user *does not want* to see any movies where a dog dies."
        "\n\n"
    )
    match steer_action:
        case "increase":
            result += "REQUEST: The user *wants* to see movies where "
        case "decrease":
            result += "REQUEST: The user *does not want* to see movies where "
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


def prompt_append_althybrid(
    tag_names: list[str],
    steer_action: str,
) -> str:
    assert len(tag_names) > 0
    # create templated LLM prompt to get a natural-sounding append sentence
    result = ""
    result += (
        "You are helping a user update how they describe their movie preferences"
        " by rephrasing their changed preferences into a single sentence."
        " This sentence will be appended to their original description."
        "\n"
        "Guidelines:"
        "\n"
        '- Write exactly *one* sentence. The sentence must start with "The user".'
        "\n"
        '- Do not include any "REQUEST" in the response. Only give the rephrased sentence.'
        "\n"
        "- Use natural language."
        "\n\n"
        "For example:"
        "\n"
        "REQUEST: The user *does not want* to see [Romance]."
        "\n"
        "The user *does not want* to see Romance."
        "\n\n"
    )
    match steer_action:
        case "increase":
            result += "REQUEST: Please show the user movies that satisfy: "
        case "decrease":
            result += "REQUEST: The user *does not want* to see "
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


def append_llm(
    profiles_original: list[TextProfile],
    steered_tags: list[TagID],
    steer_action: str,
    df_taginfo: TagInfoDF,
    tokenizer,
    model,
    template: Callable[
        [
            list[str],
            str,
        ],
        str,
    ] = default_prompt_append,
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
    Steers a profile by appending it with a single LLM-generated sentence.
    Works by prompting the LLM to rephrase a request more naturally, then appending it to the profile.

    :param profiles_original: Original profiles
    :type profiles_original: list[TextProfile]
    :param steered_tags: Tag IDs to steer
    :type steered_tags: list[TagID]
    :param steer_action: Type of steering. Must be "increase" or "decrease".
    :type steer_action: str
    :param df_taginfo: Tag metadata
    :type df_taginfo: TagInfoDF
    :param tokenizer: Model tokenizer object. Must be able to call apply_chat_template
    :param model: Some AutoModelForCausalLM-loadable object
    :param use_partial_cache: True iff partial progress should be loaded. (Saving is determined by partial_cache_fname)
    :type use_partial_cache: bool
    :param template: Function to create template LLM prompt based on a list of tag names
    :type template: Callable[[list[str], str], str]
    :param system_prompt: System prompt to use
    :type system_prompt: str
    :param seed: Random seed used for LLM prompting if not None
    :type seed: Optional[int]
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
    # NOTE: there is actually only one LLM call across all prompts here
    note_prompt_system = []
    note_prompt_main = []
    note_output_raw = []
    # Attempt loading from partial cache
    if use_partial_cache:
        assert partial_cache_fname is not None
        print(f"using partial cache: {partial_cache_fname}")
        if os.path.exists(partial_cache_fname):
            print("loading from cache")
            with open(partial_cache_fname, "rb") as f:
                (
                    steered,
                    note_prompt_system,
                    note_prompt_main,
                    note_output_raw,
                ) = pickle.load(f)
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
                pickle.dump(
                    (
                        steered,
                        note_prompt_system,
                        note_prompt_main,
                        note_output_raw,
                    ),
                    f,
                )

        # if we already created the appended item, just use it
        constructed_main_prompt: str
        if len(note_output_raw) > 0:
            constructed_main_prompt = note_prompt_main[-1]
            new_sentence = note_output_raw[-1]
        else:
            # otherwise re-create it
            constructed_main_prompt = template(
                steered_tag_names,
                steer_action,
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
            new_sentence = model_instruct_generate(
                tokenizer=tokenizer,
                model=model,
                messages=messages,
                seed=seed,
            )
        note_prompt_system.append(system_prompt)
        note_prompt_main.append(constructed_main_prompt)
        note_output_raw.append(new_sentence)
        steered.append(
            " ".join(
                [
                    p,
                    new_sentence,
                ]
            )
        )
        did_update += 1

    steered_notes = {
        "prompt_system": note_prompt_system,
        "prompt_main": note_prompt_main,
        "output_raw": note_output_raw,
    }
    assert len(steered) == len(profiles_original)
    for notes in steered_notes.values():
        assert len(notes) == len(profiles_original)
    return (steered, steered_notes)


def default_prompt_rewrite(
    tag_names: list[str],
    steer_action: str,
    original_profile: str,
) -> str:
    assert len(tag_names) > 0
    # create templated LLM prompt to rewrite a profile fully
    result = ""
    match steer_action:
        case "increase":
            result += "Modify the user profile so that the user also likes movies that satisfy: "

        case "decrease":
            result += "Modify the user profile so that the user does not like movies that satisfy: "
        case _:
            raise NotImplementedError
    if len(tag_names) > 1:
        result += "["
        result += ", ".join([f'"{e}"' for e in tag_names])
        result += "]."
    else:
        result += f"{tag_names[0]}."
    result += " Keep all the profile as similar as possible for all other preferences."
    result += "\n\n"
    # TODO few-shot examples?
    result += f"ORIGINAL PROFILE: {original_profile}"
    result += "\n"
    result += "UPDATED PROFILE:"
    return result


def prompt_rewrite_alt5_genre(
    tag_names: list[str],
    steer_action: str,
    original_profile: str,
) -> str:
    assert len(tag_names) > 0
    # create templated LLM prompt to rewrite a profile fully
    result = ""
    match steer_action:
        case "increase":
            result += "Modify the user profile to show that the user *wants* to see "

        case "decrease":
            result += "Modify the user profile to show that the user *does not want* to see "
        case _:
            raise NotImplementedError
    if len(tag_names) > 1:
        result += "["
        result += ", ".join([f'"{e}"' for e in tag_names])
        result += "]"
    else:
        result += f"{tag_names[0]}"
    result += " movies."
    result += "\n"
    result += "Keep all the profile as similar as possible for all other preferences."
    result += "\n\n"
    # TODO few-shot examples?
    result += f"ORIGINAL PROFILE: {original_profile}"
    result += "\n"
    result += "UPDATED PROFILE:"
    return result


def prompt_rewrite_alt5_ddd(
    tag_names: list[str],
    steer_action: str,
    original_profile: str,
) -> str:
    assert len(tag_names) > 0
    # create templated LLM prompt to rewrite a profile fully
    result = ""
    match steer_action:
        case "increase":
            result += "Modify the user profile to show that the user *wants* to see movies where "

        case "decrease":
            result += "Modify the user profile to show that the user *does not want* to see movies where "
        case _:
            raise NotImplementedError
    if len(tag_names) > 1:
        result += "["
        result += ", ".join([f'"{e}"' for e in tag_names])
        result += "]"
    else:
        result += f"{tag_names[0]}"
    result += "."
    result += "\n"
    result += "Keep all the profile as similar as possible for all other preferences."
    result += "\n\n"
    # TODO few-shot examples?
    result += f"ORIGINAL PROFILE: {original_profile}"
    result += "\n"
    result += "UPDATED PROFILE:"
    return result


def prompt_rewrite_althybrid(
    tag_names: list[str],
    steer_action: str,
    original_profile: str,
) -> str:
    assert len(tag_names) > 0
    # create templated LLM prompt to rewrite a profile fully
    result = ""
    match steer_action:
        case "increase":
            result += "Modify the user profile to show the user movies that satisfy: "

        case "decrease":
            result += "Modify the user profile to show that the user *does not want* to see "
        case _:
            raise NotImplementedError
    if len(tag_names) > 1:
        result += "["
        result += ", ".join([f'"{e}"' for e in tag_names])
        result += "]"
    else:
        result += f"{tag_names[0]}"
    result += "."
    result += "\n"
    result += "Keep all the profile as similar as possible for all other preferences."
    result += "\n\n"
    # TODO few-shot examples?
    result += f"ORIGINAL PROFILE: {original_profile}"
    result += "\n"
    result += "UPDATED PROFILE:"
    return result


def rewrite_llm(
    profiles_original: list[TextProfile],
    steered_tags: list[TagID],
    steer_action: str,
    df_taginfo: TagInfoDF,
    tokenizer,
    model,
    template: Callable[
        [
            list[str],
            str,
            str,
        ],
        str,
    ] = default_prompt_rewrite,
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
    Steers a profile by prompting a LLM to fully rewrite it.

    :param profiles_original: Original profiles
    :type profiles_original: list[TextProfile]
    :param steered_tags: Tag IDs to steer
    :type steered_tags: list[TagID]
    :param steer_action: Type of steering. Must be "increase" or "decrease".
    :type steer_action: str
    :param df_taginfo: Tag metadata
    :type df_taginfo: TagInfoDF
    :param tokenizer: Model tokenizer object. Must be able to call apply_chat_template
    :param model: Some AutoModelForCausalLM-loadable object
    :param use_partial_cache: True iff partial progress should be loaded. (Saving is determined by partial_cache_fname)
    :type use_partial_cache: bool
    :param template: Function to create template LLM prompt based on a list of tag names and the full original profile text.
    :type template: Callable[[list[str], str, str], str]
    :param system_prompt: System prompt to use
    :type system_prompt: str
    :param seed: Random seed used for LLM prompting if not None
    :type seed: Optional[int]
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
                    steered,
                    note_prompt_system,
                    note_prompt_main,
                ) = pickle.load(f)
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
                pickle.dump(
                    (
                        steered,
                        note_prompt_system,
                        note_prompt_main,
                    ),
                    f,
                )

        constructed_main_prompt = template(
            steered_tag_names,
            steer_action,
            p,
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
        new_profile = model_instruct_generate(
            tokenizer=tokenizer,
            model=model,
            messages=messages,
            seed=seed,
        )
        note_prompt_system.append(system_prompt)
        note_prompt_main.append(constructed_main_prompt)
        steered.append(new_profile)
        did_update += 1

    steered_notes = {
        "prompt_system": note_prompt_system,
        "prompt_main": note_prompt_main,
    }
    assert len(steered) == len(profiles_original)
    for notes in steered_notes.values():
        assert len(notes) == len(profiles_original)
    return (steered, steered_notes)
