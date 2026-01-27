import os
import pickle
import random
from typing import Optional, Tuple

import pandas as pd
from tqdm import tqdm

from custom_types import (
    AddtNotesDict,
    ItemID,
    RankingDF,
    SampleInfo,
    TagInfoDF,
    TagLinksDF,
    TextProfile,
)
from misc.item_template import get_tag_names


def naive_random(
    per_user_samples: list[SampleInfo],
    rng: Optional[random.Random] = None,
    seed: Optional[int] = None,
) -> Tuple[
    list[RankingDF],
    AddtNotesDict,
]:
    """
    Rank items randomly.

    :param per_user_samples: Experiment samples
    :type per_user_samples: list[SampleInfo]
    :param rng: Random object to use. Takes priority over seed.
    :type rng: Optional[random.Random]
    :param seed: Seed for random object to use.
    :type seed: Optional[int]
    :return: List of rankings, and dict containing extra notes per entry
    :rtype: Tuple[list[RankingDF], AddtNotesDict]
    """
    # create randomizer
    rng_used = random.Random(seed)
    if rng is not None:
        rng_used = rng

    # Create the rankings results
    rankings: list[RankingDF] = []
    # nothing to cache-load; this is a quick operation

    # now create the rest of the rankings as needed
    for u in tqdm(per_user_samples[len(rankings) :]):
        # nothing to cache-save; this is a quick operation
        # generate random scores per item
        randomized_scores: list[float] = list(range(len(u.willrank_ids)))
        randomized_scores = [e / max(randomized_scores) for e in randomized_scores]
        rng_used.shuffle(randomized_scores)
        u_ranking: RankingDF = pd.DataFrame(
            {
                "id_item": u.willrank_ids,
                "score": randomized_scores,
            }
        )
        u_ranking = u_ranking.sort_values(
            by="score",
            ascending=False,
            na_position="last",
        )
        rankings.append(u_ranking)

    rankings_notes: AddtNotesDict = {}
    assert len(rankings) == len(per_user_samples)
    for notes in rankings_notes.values():
        assert len(notes) == len(per_user_samples)
    return (rankings, rankings_notes)


def naive_tagcount(
    per_user_samples: list[SampleInfo],
    profiles: list[TextProfile],
    df_taginfo: TagInfoDF,
    df_taglinks: TagLinksDF,
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
    Rank items by counting how many tags they share with the profile's tag list.
    Assumes that profiles are in naive "tag list" format.

    :param per_user_samples: Experiment samples
    :type per_user_samples: list[SampleInfo]
    :param profiles: User profiles. Assumes that profiles are in "tag list" format.
    :type profiles: list[TextProfile]
    :param df_taginfo: Tag metadata
    :type df_taginfo: TagInfoDF
    :param df_taglinks: Tag links data
    :type df_taglinks: TagLinksDF
    :param rng: Random object to use. Takes priority over seed.
    :type rng: Optional[random.Random]
    :param seed: Seed for random object to use.
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
    note_tag_overlap_tally: list[int] = []
    # Attempt loading from partial cache
    if use_partial_cache:
        assert partial_cache_fname is not None
        print(f"using partial cache: {partial_cache_fname}")
        if os.path.exists(partial_cache_fname):
            print("loading from cache")
            with open(partial_cache_fname, "rb") as f:
                (
                    rankings,
                    note_tag_overlap_tally,
                ) = pickle.load(f)
            print(f"already existing: {len(rankings)} rankings")
        else:
            print("cache not yet existing; will create")

    # now create the rest of the rankings as needed
    did_update = 0
    for u, p in tqdm(
        zip(
            per_user_samples[len(rankings) :],
            profiles[len(rankings) :],
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
                        note_tag_overlap_tally,
                    ),
                    f,
                )

        # get the profile tag collection
        profile_tag_names = p.split("\n")
        # get tags for each rankable item
        item_tag_ids = [
            df_taglinks[
                (df_taglinks["relevant"] == True) & (df_taglinks["id_item"] == item_id)
            ]["id_tag"].tolist()
            for item_id in u.willrank_ids
        ]
        item_tag_names = [
            get_tag_names(
                tag_ids=l,
                df_taginfo=df_taginfo,
            )
            for l in item_tag_ids
        ]
        # count how many overlaps there are between each item and profile
        item_overlap_scores = [
            (
                i,
                len(set(l) & set(profile_tag_names)),
            )
            for i, l in zip(
                u.willrank_ids,
                item_tag_names,
            )
        ]
        # because there is a high chance that we will need to tiebreak,
        # manually construct ranking from these scores
        score_bins: dict[int, list[ItemID]] = {}
        for i, s in item_overlap_scores:
            if s not in score_bins:
                score_bins[s] = []
            score_bins[s].append(i)
        ranking_manual_items: list[ItemID] = []
        ranking_manual_scores: list[float] = []
        for k in sorted(
            list(score_bins.keys()),
            reverse=True,
        ):
            ranking_manual_items += score_bins[k]
            # tiebreaking noise
            ranking_manual_scores += [
                k + (rng_used.random() * 0.9) for _ in score_bins[k]
            ]
        u_ranking: RankingDF = pd.DataFrame(
            {
                "id_item": ranking_manual_items,
                "score": ranking_manual_scores,
            }
        )
        u_ranking = u_ranking.sort_values(
            by="score",
            ascending=False,
            na_position="last",
        )
        rankings.append(u_ranking)
        did_update += 1

    rankings_notes: AddtNotesDict = {
        "tag_overlap_tally": note_tag_overlap_tally,
    }
    assert len(rankings) == len(per_user_samples)
    for notes in rankings_notes.values():
        assert len(notes) == len(per_user_samples)
    return (rankings, rankings_notes)


def naive_oracle(
    per_user_samples: list[SampleInfo],
    prioritize_accuracy: bool = False,
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
    Rank items by putting all accurate and targeted items at the top, and all non-accurate and non-targeted items at the bottom.
    Items that are accurate XOR targeted are ordered depending on whether we prioritize accuracy or not.
    Items within each of these groups are randomly ordered; only the larger ranking has strict group boundaries.

    :param per_user_samples: Experiment samples
    :type per_user_samples: list[SampleInfo]
    :param prioritize_accuracy: True iff accurate but non-targeted items are ranked above non-accurate but targeted items.
    :type prioritize_accuracy: bool
    :param rng: Random object to use. Takes priority over seed.
    :type rng: Optional[random.Random]
    :param seed: Seed for random object to use.
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
    assert partial_cache_incr > 0

    # create randomizer
    rng_used = random.Random(seed)
    if rng is not None:
        rng_used = rng

    # Create the rankings results
    rankings: list[RankingDF] = []
    # Attempt loading from partial cache
    if use_partial_cache:
        assert partial_cache_fname is not None
        print(f"using partial cache: {partial_cache_fname}")
        if os.path.exists(partial_cache_fname):
            print("loading from cache")
            with open(partial_cache_fname, "rb") as f:
                (rankings,) = pickle.load(f)
            print(f"already existing: {len(rankings)} rankings")
        else:
            print("cache not yet existing; will create")

    # now create the rest of the rankings as needed
    did_update = 0
    for u in tqdm(per_user_samples[len(rankings) :]):
        if (
            did_update > 0
            and did_update % partial_cache_incr == 0
            and (partial_cache_fname is not None)
        ):
            print(f"updating cache at increment {did_update}")
            with open(partial_cache_fname, "wb") as f:
                pickle.dump((rankings,), f)

        ordered_ids: list[ItemID] = []
        # put acc+rel first in shuffled order
        acc_rel = list(set(u.upcoming_ids) & set(u.targeted_ids))
        rng_used.shuffle(acc_rel)
        ordered_ids += acc_rel
        # if we strictly prioritize accuracy, we put the acc+nonrel before nonacc+rel
        # otherwise put nonacc+rel first
        acc_nonrel = list(set(u.upcoming_ids) - set(ordered_ids))
        rng_used.shuffle(acc_nonrel)
        nonacc_rel = list(set(u.targeted_ids) - set(ordered_ids))
        rng_used.shuffle(nonacc_rel)
        assert len(set(acc_nonrel) & set(nonacc_rel)) == 0
        if prioritize_accuracy:
            ordered_ids += acc_nonrel
            ordered_ids += nonacc_rel
        else:
            ordered_ids += nonacc_rel
            ordered_ids += acc_nonrel
        # finally put everything else
        nonacc_nonrel = list(set(u.willrank_ids) - set(ordered_ids))
        rng_used.shuffle(nonacc_nonrel)
        ordered_ids += nonacc_nonrel
        # generate descending scores
        descending_scores: list[float] = list(reversed(range(len(u.willrank_ids))))
        u_ranking: RankingDF = pd.DataFrame(
            {
                "id_item": ordered_ids,
                "score": descending_scores,
            }
        )
        u_ranking = u_ranking.sort_values(
            by="score",
            ascending=False,
            na_position="last",
        )
        rankings.append(u_ranking)
        did_update += 1

    rankings_notes: AddtNotesDict = {}
    assert len(rankings) == len(per_user_samples)
    for notes in rankings_notes.values():
        assert len(notes) == len(per_user_samples)
    return (rankings, rankings_notes)
