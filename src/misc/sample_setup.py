import math
import random
from typing import Optional, Tuple

import pandas as pd
from tqdm import tqdm

from custom_types import (
    ItemID,
    ItemInfoDict,
    RatingDataDF,
    TagID,
    SampleInfo,
    UserID,
    validate_rating_data_df,
    validate_tag_info_df,
    validate_tag_links_df,
)
from custom_types import TagLinksDF


def filter_selectable_tags(
    df_taglinks: TagLinksDF,
    tag_strict_range: Tuple[float, float] = (0, 0.5),
    tag_subprefer_range: Tuple[float, float] = (0.9, 1),
) -> list[ItemID]:
    # identify what fraction of the dataset each tag is relevant to
    df_tag_fractions = df_taglinks.groupby("id_tag")["relevant"]
    df_tag_fractions = df_tag_fractions.mean()
    # filter on the strict relevance range
    df_tag_fractions = df_tag_fractions[
        (df_tag_fractions >= tag_strict_range[0])
        & (df_tag_fractions <= tag_strict_range[1])
    ].sort_values(ascending=False)
    # get preferred percentile of tag relevance within that range
    ix_low = math.floor(len(df_tag_fractions) * (1 - tag_subprefer_range[1]))
    ix_high = math.ceil(len(df_tag_fractions) * (1 - tag_subprefer_range[0])) + 1
    tags_filtered = df_tag_fractions.iloc[ix_low:ix_high].index.tolist()
    return tags_filtered


def sample_setup(
    df_taglinks: TagLinksDF,
    df_ratings: RatingDataDF,
    item_info: ItemInfoDict,
    focus_tags: Optional[list[TagID]] = None,
    tag_count: int = 1,
    tag_strict_range: Tuple[float, float] = (0, 0.5),
    tag_subprefer_range: Tuple[float, float] = (0.9, 1),
    user_count: int = 10,
    user_skip_n: int = 0,
    focus_users: Optional[list[UserID]] = None,
    user_min_ratings: int = 0,
    user_history_n: int = 20,
    user_upcoming_n: int = 1,
    draw_num: Tuple[int, int] = (10, 50),
    force_exclude_items: list[ItemID] = [],
    strict_separate_upcoming: bool = True,
    rng: Optional[random.Random] = None,
    seed: Optional[int] = None,
) -> Tuple[
    list[TagID],
    list[SampleInfo],
]:
    """
    Generates a randomly selected steering tag, then samples users, user histories, and ranking pools involving a mix of non/target and non/relevant items for each user.

    :param df_taglinks: Information about what tags are linked to items
    :type df_taglinks: TagLinksDF
    :param df_ratings: Information about user rating history
    :type df_ratings: RatingDataDF
    :param item_info: Information about items
    :type item_info: ItemInfoDict
    :param focus_tags: Tags to focus on in the sampling. Takes priority over tag_count
    :type focus_tags: Optional[list[TagID]]
    :param tag_count: Number of tags to sample steering for
    :type tag_count: int
    :param tag_strict_range: (X,Y), restrict ourselves to tags between X and Y fraction popularity. Must be (0,1) at widest.
    :type tag_strict_range: Tuple[float, float]
    :param tag_subprefer_range: (X,Y), pick from the post-strict-filter pool some random tag(s) between top X and Y percentile popularity. Must be (0,1) at widest.
    :type tag_subprefer_range: Tuple[float, float]
    :param user_count: number of users to sample max. Negative means all.
    :type user_count: int
    :param user_skip_n: number of users to skip sampling. 0 means none. Must be non-negative.
    :type user_skip_n: int
    :param focus_users: IDs of users to focus on. Overrides user_count. Still subject to all other user requirements.
    :type focus_users: Optional[list[UserID]]
    :param user_min_ratings: Minimum # of total ratings needed for a user to be sampleable
    :type user_min_ratings: int
    :param user_history_n: number of items to use in user history. Must be non-negative.
    :type user_history_n: int
    :param user_upcoming_n: number of items to include in user upcoming prediction. Must be non-negative.
    :type user_upcoming_n: int
    :param draw_num: (X,Y), include for ranking exactly X items unrelated to selected tags and exactly Y items related to selected tags. Negative means "draw as many as possible".
    :type draw_num: Tuple[int, int]
    :param force_exclude_items: Items to never include for ranking. May still be used for history.
    :type force_exclude_items: list[ItemID]
    :param strict_separate_upcoming: True iff "non-upcoming" implies "was literally never rated by the user in the full dataset"
    :type strict_separate_upcoming: bool
    :param rng: Random object to use. Takes priority over seed.
    :type rng: Optional[random.Random]
    :param seed: Seed for random object to use.
    :type seed: Optional[int]
    :return: Tuple of selected tags, full experiment sampled IDs per user
    :rtype: Tuple[list[TagID], list[SampleInfo]]
    """
    # input checks
    validate_rating_data_df(df_ratings)
    validate_tag_links_df(df_taglinks)
    assert tag_count >= 1
    assert tag_strict_range[0] <= tag_strict_range[1]
    assert tag_strict_range[0] >= 0
    assert tag_strict_range[1] <= 1
    assert tag_subprefer_range[0] <= tag_subprefer_range[1]
    assert tag_subprefer_range[0] >= 0
    assert tag_subprefer_range[1] <= 1
    assert user_skip_n >= 0
    assert user_history_n >= 0
    assert user_upcoming_n >= 0

    # create randomizer
    rng_used = random.Random(seed)
    if rng is not None:
        rng_used = rng

    # determine what tags we are focusing on, if not already given
    selected_tags: list[TagID] = []
    if focus_tags is not None:
        # if we already have a list of focused tags, use those
        selected_tags = [e for e in focus_tags]
    else:
        # otherwise, pick a tag based on the filtering criteria
        usable_tags = filter_selectable_tags(
            df_taglinks=df_taglinks,
            tag_strict_range=tag_strict_range,
            tag_subprefer_range=tag_subprefer_range,
        )
        assert len(usable_tags) >= tag_count
        selected_tags = rng_used.sample(usable_tags, tag_count)
    print(f"Selected tag(s): {selected_tags}")

    # first identify all selected-tag-relevant and nonrelevant items
    steer_target_ids: list[ItemID] = (
        df_taglinks[
            (df_taglinks["relevant"] == True)
            & (df_taglinks["id_tag"].isin(selected_tags))
        ]["id_item"]
        .dropna()
        .unique()
        .tolist()
    )
    # TODO improve handling of less-known items
    # (e.g. not marked as relevant=false but also not marked as relevant=true)
    # depending on dataset format
    steer_nontarget_ids: list[ItemID] = (
        df_taglinks[
            (df_taglinks["relevant"] == False)
            & (df_taglinks["id_tag"].isin(selected_tags))
        ]["id_item"]
        .dropna()
        .unique()
        .tolist()
    )
    # remove force excluded items
    steer_target_ids = list(set(steer_target_ids) - set(force_exclude_items))
    steer_nontarget_ids = list(set(steer_nontarget_ids) - set(force_exclude_items))
    # print log info
    print(f"total usable relevant: {len(steer_target_ids)}")
    print(f"total usable not-relevant: {len(steer_nontarget_ids)}")
    print(f"total items force excluded: {len(force_exclude_items)}")

    # sample users and rating pools...
    per_user_samples: list[SampleInfo] = []
    samples_skipped: list[SampleInfo] = []
    # set up logging for user data scanning...
    skip_reasons = {
        "forcibly_skipped": 0,
        "already_enough": 0,
        "non_focus": 0,
        "insufficient_total_ratings": 0,
        "no_history": 0,
        "no_future": 0,
        "future_exceeds_target": 0,
        "future_exceeds_nontarget": 0,
        "insufficient_target": 0,
        "insufficient_nontarget": 0,
    }
    num_users_total = len(df_ratings.groupby("id_user"))
    scanned_incr = 10
    if num_users_total > 100:
        scanned_incr = int(num_users_total / 20)
    scanned_total = 0
    # start building user steering information
    df_ratings = df_ratings.sort_values(
        by="time",
        ascending=True,
        na_position="last",
        kind="stable",  # avoiding nondeterministic behavior
    )
    # NOTE: arbitrary threshold for whether we consider an item
    # actually within a user's interactions?
    # df_ratings_filtered = df_ratings[df_ratings["score"] >= 2.5]
    df_ratings_filtered = df_ratings[df_ratings["score"] >= 0]
    # iterate through users...
    for id_user, df_user_ratings in tqdm(
        df_ratings_filtered.groupby(
            "id_user",
            sort=False,
        )
    ):
        # do logging output
        if scanned_total > 0 and scanned_total % scanned_incr == 0:
            print(
                f"{scanned_total}/{num_users_total} scanned "
                + f"({100*scanned_total/num_users_total}%); "
                + f"skip reasons? {skip_reasons}"
            )
        scanned_total += 1
        # skip extra work if we have already obtained enough
        if (user_count >= 0) and (focus_users is None):
            if len(per_user_samples) >= user_count:
                # we already have enough
                skip_reasons["already_enough"] += 1
                continue
        # remove non-focused users
        if focus_users is not None:
            if id_user not in focus_users:
                # user ID is not within the focus pool
                # move on to the next candidate
                skip_reasons["non_focus"] += 1
                continue
        # check if user has interacted with enough items at all
        if len(df_user_ratings) < (user_min_ratings):
            skip_reasons["insufficient_total_ratings"] += 1
            continue
        if len(df_user_ratings) < (user_history_n):
            skip_reasons["no_history"] += 1
            continue
        if len(df_user_ratings) < (user_history_n + user_upcoming_n):
            skip_reasons["no_future"] += 1
            continue
        # start sampling items
        user_history = list(df_user_ratings["id_item"])
        user_scores = list(df_user_ratings["score"])
        # extract user's previous items for profile creation
        user_history_used = user_history[:user_history_n]
        user_scores_used = user_scores[:user_history_n]
        # identify all possible sample-able item IDs
        eligible_target_ids = list(set(steer_target_ids) - set(user_history_used))
        eligible_nontarget_ids = list(set(steer_nontarget_ids) - set(user_history_used))
        # sample items to rank
        willrank_ids: list[ItemID] = []
        upcoming_ids: list[ItemID] = []
        targeted_ids: list[ItemID] = []
        # prioritize upcoming items
        drawn_already = [0, 0]  # count of [target, nontarget]
        if user_upcoming_n > 0:
            # we do want to draw next-ranked items?
            upcoming_ids += user_history[
                user_history_n : user_history_n + user_upcoming_n
            ]
            willrank_ids += upcoming_ids
            # update the drawn counts
            # if we have too many of either, cancel the operation
            targeted_ids += [e for e in upcoming_ids if (e in eligible_target_ids)]
            drawn_already[0] += len(targeted_ids)
            if draw_num[0] >= 0 and drawn_already[0] > draw_num[0]:
                skip_reasons["future_exceeds_target"] += 1
                continue
            drawn_already[1] += len(upcoming_ids) - len(targeted_ids)
            if draw_num[1] >= 0 and drawn_already[1] > draw_num[1]:
                skip_reasons["future_exceeds_nontarget"] += 1
                continue
        assert len(willrank_ids) == (drawn_already[0] + drawn_already[1])
        # update eligible pools to exclude possible upcoming items?
        if strict_separate_upcoming:
            # remove all possible items that the user interacted with
            eligible_target_ids = list(set(eligible_target_ids) - set(user_history))
            eligible_nontarget_ids = list(
                set(eligible_nontarget_ids) - set(user_history)
            )
        else:
            # we're fine with including some items the user interacted with
            # after the "next upcoming" window
            eligible_target_ids = list(set(eligible_target_ids) - set(upcoming_ids))
            eligible_nontarget_ids = list(
                set(eligible_nontarget_ids) - set(upcoming_ids)
            )
        # draw remaining appropriate items from the not-upcoming pool
        assert (draw_num[0] < 0) or (draw_num[0] - drawn_already[0] >= 0)
        assert (draw_num[1] < 0) or (draw_num[1] - drawn_already[1] >= 0)
        # draw steering-targeted items
        sampled = eligible_target_ids
        if draw_num[0] > 0:
            # if we need to limit the number of sampled targets somehow
            needed = draw_num[0] - drawn_already[0]
            if needed > len(eligible_target_ids):
                skip_reasons["insufficient_target"] += 1
                continue
            # TODO this assumes there is exactly one upcoming item
            # TODO this assumes that item data contains popularity
            ref_popularity = [item_info[e]["popularity"] for e in willrank_ids][0]
            # TODO: this targets items that are closest in popularity to
            # THIS ONE SINGLE USER's next item
            # But, because there are usually fewer targeted items than
            # non-targeted items, JZ suspects that there will be much more
            # variance in popularity within the targeted set,
            # than the non-targeted set
            sample_popularity_diff = [
                (e, abs(item_info[e]["popularity"] - ref_popularity))
                for e in list(eligible_target_ids)
            ]
            sample_popularity_diff.sort(key=lambda x: x[1])
            sampled = [x[0] for x in sample_popularity_diff[:needed]]
            # sampled = rng.sample(
            #     eligible_target_ids,
            #     k=needed,
            # )
        willrank_ids += sampled
        targeted_ids += sampled
        # draw steering-nontargeted items
        sampled = eligible_nontarget_ids
        if draw_num[1] > 0:
            # if we need to limit the number of sampled nontargets somehow
            needed = draw_num[1] - drawn_already[1]
            if needed > len(eligible_nontarget_ids):
                skip_reasons["insufficient_nontarget"] += 1
                continue
            # TODO this assumes there is exactly one upcoming item
            # TODO this assumes that item data contains popularity
            ref_popularity = [item_info[e]["popularity"] for e in willrank_ids][0]
            # TODO: this targets items that are closest in popularity to
            # THIS ONE SINGLE USER's next item
            # see above note for this
            sample_popularity_diff = [
                (e, abs(item_info[e]["popularity"] - ref_popularity))
                for e in list(eligible_nontarget_ids)
            ]
            sample_popularity_diff.sort(key=lambda x: x[1])
            sampled = [x[0] for x in sample_popularity_diff[:needed]]
            # sampled = rng.sample(
            #     eligible_nontarget_ids,
            #     k=needed,
            # )
        willrank_ids += sampled
        for i in user_history_used:
            assert i not in willrank_ids
        for i in upcoming_ids + targeted_ids:
            assert i in willrank_ids
        if draw_num[0] >= 0 and draw_num[1] >= 0:
            assert len(willrank_ids) == (draw_num[0] + draw_num[1])
        new_sample = SampleInfo(
            user_id=id_user,  # type: ignore
            history_ids=user_history_used,
            history_scores=user_scores_used,
            willrank_ids=willrank_ids,
            upcoming_ids=upcoming_ids,
            targeted_ids=targeted_ids,
        )
        if len(samples_skipped) < user_skip_n:
            samples_skipped.append(new_sample)
            skip_reasons["forcibly_skipped"] += 1
            continue
        else:
            per_user_samples.append(new_sample)
    print("done iterating through users for tag+item sampling!")
    print(f"collected {len(per_user_samples)} samples")
    print(skip_reasons)
    return (selected_tags, per_user_samples)
