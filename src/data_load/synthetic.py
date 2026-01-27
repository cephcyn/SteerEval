import random
from typing import Optional, Tuple

import pandas as pd
from custom_types import (
    ItemInfoDict,
    TagInfoDF,
    TagLinksDF,
    validate_tag_info_df,
    validate_tag_links_df,
)


def random_tag_genres(
    item_info: ItemInfoDict,
    random_rate: float = 0.5,
    rng: Optional[random.Random] = None,
    seed: Optional[int] = None,
) -> Tuple[
    TagInfoDF,
    TagLinksDF,
]:
    # TODO docstring
    # create randomizer
    rng_used = random.Random(seed)
    if rng is not None:
        rng_used = rng

    # there exists one tag and it is random
    id_tag = 1
    dict_taginfo = {
        "id_tag": [id_tag],
        "name": [f"RandomTag_p={random_rate}"],
    }
    dict_taglinks = {
        "id_item": [],
        "id_tag": [],
        "relevant": [],
        "score_relevant": [],
        "score_known": [],
    }
    # apply the randomness
    for id_item in item_info.keys():
        random_relevance = rng_used.choice([True, False])
        dict_taglinks["id_item"].append(id_item)
        dict_taglinks["id_tag"].append(id_tag)
        dict_taglinks["relevant"].append(random_relevance)
        dict_taglinks["score_relevant"].append(1 if random_relevance else 0)
        dict_taglinks["score_known"].append(1)
    # convert to output
    df_taginfo = pd.DataFrame(dict_taginfo)
    df_taglinks = pd.DataFrame(dict_taglinks)

    # verify that output matches format minimum
    validate_tag_info_df(df_taginfo)
    validate_tag_links_df(df_taglinks)
    # return results
    return df_taginfo, df_taglinks
