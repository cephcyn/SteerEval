from typing import Optional
from custom_types import (
    ItemID,
    ItemInfoDict,
    RatingDataDF,
    TagID,
    TagInfoDF,
    TagLinksDF,
    UserID,
)


def get_tag_names(
    tag_ids: list[TagID],
    df_taginfo: TagInfoDF,
) -> list[str]:
    """
    Transform a list of tag IDs into a set of tag names.

    :param tag_ids: List of tag IDs
    :type tag_ids: list[TagID]
    """
    tag_ids = list(set(tag_ids))
    tag_strs = []
    if len(tag_ids) > 0:
        tag_strs = [
            df_taginfo[df_taginfo["id_tag"] == e].iloc[0]["name"] for e in tag_ids
        ]
    return tag_strs


def templatify_item_id(
    metadata: str,
    user_id: UserID,
    item_id: ItemID,
    df_ratings: RatingDataDF,
    df_taglinks: TagLinksDF,
    df_taginfo: TagInfoDF,
    item_info: ItemInfoDict,
    use_score: bool,
    targeted_tag_ids: Optional[list[TagID]] = None,
) -> str:
    """
    Transform a given user+item combination into a string consisting of
    metadata for the user's interaction with that item.
    Format consists of multiple lines, each formatted like `FieldName: FieldContents`.

    :param metadata: Data to include in the string.
    Currently supports "all_tags", "target_tags", "title", "description", "title+description"
    :type metadata: str
    :param user_id: User ID
    :type user_id: UserID
    :param item_id: Item ID
    :type item_id: ItemID
    :param df_ratings: Rating data
    :type df_ratings: RatingDataDF
    :param df_taglinks: Tag data
    :type df_taglinks: TagLinksDF
    :param df_taginfo: Tag metadata
    :type df_taginfo: TagInfoDF
    :param item_info: Item metadata
    :type item_info: ItemInfoDict
    :param use_score: Whether to include the user score rating
    :type use_score: bool
    :param targeted_tag_id: Targeted tag ID, if metadata="target_tags"
    :type targeted_tag_id: Optional[list[TagID]]
    :return: Description
    :rtype: str
    """
    rating = -1
    if use_score:
        rating = df_ratings[
            (df_ratings["id_user"] == user_id) & (df_ratings["id_item"] == item_id)
        ].iloc[0]["score"]
    output_lines = []
    metadata_components = metadata.split("+")
    for m in metadata_components:
        match m:
            case "all_tags":
                relevant_tags = df_taglinks[
                    (df_taglinks["relevant"] == True)
                    & (df_taglinks["id_item"] == item_id)
                ]["id_tag"].tolist()
                relevant_tag_names = [
                    df_taginfo[df_taginfo["id_tag"] == e].iloc[0]["name"]
                    for e in relevant_tags
                ]
                tag_str = ", ".join(relevant_tag_names)
                output_lines += [
                    f"Movie Tags: [{tag_str}]",
                ]
            case "target_tags":
                assert targeted_tag_ids is not None
                assert len(targeted_tag_ids) > 0
                relevant_tags = df_taglinks[
                    (df_taglinks["id_tag"].isin(targeted_tag_ids))
                    & (df_taglinks["relevant"] == True)
                    & (df_taglinks["id_item"] == item_id)
                ]["id_tag"].tolist()
                relevant_tag_names = [
                    df_taginfo[df_taginfo["id_tag"] == e].iloc[0]["name"]
                    for e in relevant_tags
                ]
                tag_str = ", ".join(relevant_tag_names)
                output_lines += [
                    f"Movie Tags: [{tag_str}]",
                ]
            case "title":
                item_title = item_info[item_id]["title"]
                output_lines += [
                    f"Movie Title: {item_title}",
                ]
            case "description":
                item_desc = item_info[item_id]["description"]
                output_lines += [
                    f"Movie Description: {item_desc}",
                ]
            case _:
                raise NotImplementedError
    if use_score:
        output_lines += [
            f"User Rating: {rating}",
        ]
    return "\n".join(output_lines)


def templatify_item_id_oneline(
    metadata: str,
    user_id: UserID,
    item_id: ItemID,
    df_ratings: RatingDataDF,
    df_taglinks: TagLinksDF,
    df_taginfo: TagInfoDF,
    item_info: ItemInfoDict,
    use_score: bool,
    targeted_tag_ids: Optional[list[TagID]] = None,
) -> str:
    """
    Transform a given user+item combination into a ONE-LINE string consisting of
    metadata for the user's interaction with that item.
    Format consists of multiple lines, each formatted like `FieldName: FieldContents`.

    :param metadata: Data to include in the string.
    Currently supports "all_tags", "target_tags", "title", "description", "title+description"
    :type metadata: str
    :param user_id: User ID
    :type user_id: UserID
    :param item_id: Item ID
    :type item_id: ItemID
    :param df_ratings: Rating data
    :type df_ratings: RatingDataDF
    :param df_taglinks: Tag data
    :type df_taglinks: TagLinksDF
    :param df_taginfo: Tag metadata
    :type df_taginfo: TagInfoDF
    :param item_info: Item metadata
    :type item_info: ItemInfoDict
    :param use_score: Whether to include the user score rating
    :type use_score: bool
    :param targeted_tag_id: Targeted tag ID, if metadata="target_tags"
    :type targeted_tag_id: Optional[list[TagID]]
    :return: Description
    :rtype: str
    """
    rating = -1
    if use_score:
        rating = df_ratings[
            (df_ratings["id_user"] == user_id) & (df_ratings["id_item"] == item_id)
        ].iloc[0]["score"]
    output = ""
    match metadata:
        case "all_tags":
            relevant_tags = df_taglinks[
                (df_taglinks["relevant"] == True) & (df_taglinks["id_item"] == item_id)
            ]["id_tag"].tolist()
            relevant_tag_names = [
                df_taginfo[df_taginfo["id_tag"] == e].iloc[0]["name"]
                for e in relevant_tags
            ]
            tag_str = ", ".join(relevant_tag_names)
            output += f"tags: [{tag_str}]"
        case "target_tags":
            assert targeted_tag_ids is not None
            assert len(targeted_tag_ids) > 0
            relevant_tags = df_taglinks[
                (df_taglinks["id_tag"].isin(targeted_tag_ids))
                & (df_taglinks["relevant"] == True)
                & (df_taglinks["id_item"] == item_id)
            ]["id_tag"].tolist()
            relevant_tag_names = [
                df_taginfo[df_taginfo["id_tag"] == e].iloc[0]["name"]
                for e in relevant_tags
            ]
            tag_str = ", ".join(relevant_tag_names)
            output += f"tags: [{tag_str}]"
        case "title":
            item_title = item_info[item_id]["title"]
            output += f"{item_title}"
        case "description":
            item_desc = item_info[item_id]["description"]
            output += f'description: "{item_desc}"'
        case "title+description":
            item_title = item_info[item_id]["title"]
            item_desc = item_info[item_id]["description"]
            output += f'{item_title}: "{item_desc}"'
        case _:
            raise NotImplementedError
    if use_score:
        output += f"(User Rating: {rating})"
    return output
