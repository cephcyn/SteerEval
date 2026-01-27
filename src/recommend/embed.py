import os
import pickle
from typing import Optional, Tuple

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from custom_types import (
    AddtNotesDict,
    ItemInfoDict,
    RankingDF,
    RatingDataDF,
    SampleInfo,
    TagID,
    TagInfoDF,
    TagLinksDF,
    TextProfile,
)
from misc.item_template import templatify_item_id


def embed_similarity(
    per_user_samples: list[SampleInfo],
    profiles: list[TextProfile],
    item_data: str,
    df_ratings: RatingDataDF,
    item_info: ItemInfoDict,
    df_taginfo: TagInfoDF,
    df_taglinks: TagLinksDF,
    model: SentenceTransformer,
    targeted_tag_ids: Optional[list[TagID]] = None,
    use_partial_cache: bool = False,
    partial_cache_incr: int = 10,
    partial_cache_fname: Optional[str] = None,
) -> Tuple[
    list[RankingDF],
    AddtNotesDict,
]:
    """
    Rank items by computing embedding model similarity between profile and item metadata blurb.

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
    :param model: Text embedding model
    :param targeted_tag_ids: Targeted tag ID, if item_data="target_tags"
    :type targeted_tag_ids: Optional[list[TagID]]
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

    # Create the rankings results
    rankings: list[RankingDF] = []
    note_item_texts: list[list[str]] = []
    # Attempt loading from partial cache
    if use_partial_cache:
        assert partial_cache_fname is not None
        print(f"using partial cache: {partial_cache_fname}")
        if os.path.exists(partial_cache_fname):
            print("loading from cache")
            with open(partial_cache_fname, "rb") as f:
                (
                    rankings,
                    note_item_texts,
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
                        note_item_texts,
                    ),
                    f,
                )

        # get item texts
        item_texts = [
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
            )
            for e in u.willrank_ids
        ]
        # get embeddings
        embeddings = model.encode([p] + item_texts)
        similarities = (
            model.similarity(embeddings[0:1], embeddings[1:]).squeeze(0).tolist()
        )
        u_ranking: RankingDF = pd.DataFrame(
            {
                "id_item": u.willrank_ids,
                "score": similarities,
            }
        )
        u_ranking = u_ranking.sort_values(
            by="score",
            ascending=False,
            na_position="last",
        )
        note_item_texts.append(item_texts)
        rankings.append(u_ranking)
        did_update += 1

    rankings_notes: AddtNotesDict = {
        "item_texts": note_item_texts,
    }
    assert len(rankings) == len(per_user_samples)
    for notes in rankings_notes.values():
        assert len(notes) == len(per_user_samples)
    return (rankings, rankings_notes)
