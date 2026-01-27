import os
from typing import Any, Tuple
from pathlib import Path

import pandas as pd

from custom_types import TagInfoDF, validate_tag_info_df
from custom_types import TagLinksDF, validate_tag_links_df


def generated_reviews_tags(
    raw_dir_path: Path,
    csv_fname: str = "output.csv",
    **_: Any,
) -> Tuple[
    TagInfoDF,
    TagLinksDF,
]:
    """
    Load movie review annotation tag dataset
    NOTE: this function has not been used or tested; may be buggy

    :param raw_dir_path: Directory to read annotation data files from
    :type raw_dir_path: Path
    :param csv_fname: Name of annotation csv file
    :type csv_fname: str
    :param _: Overflow args
    :type _: Any
    :return: Tuple of tag and tag-link information
    :rtype: Tuple[TagInfoDF, TagLinksDF]
    """
    # retrieve custom annotated output csvs
    print(f"Loading file from {os.path.join(raw_dir_path, csv_fname)}")
    annots_df = pd.read_csv(os.path.join(raw_dir_path, csv_fname))

    # Do our custom reformatting now
    mapping_tags = {}
    mapping_tagdata = {}
    for x, r in annots_df.iterrows():
        annot_lines = r["model_annotated"].split("\n")
        annot_lines = [e.strip().lstrip("-").strip() for e in annot_lines]
        annot_lines = [e for e in annot_lines if len(e) > 0]
        for annot in annot_lines:
            # extract tag body from line
            annot_body = [e.strip() for e in annot.strip().split(" ")]
            annot_body = [e for e in annot_body if len(e) > 0]
            if len(annot_body) < 1:
                continue
            if annot_body[-1][0] == "(" and annot_body[-1][-1] == ")":
                annot_body = annot_body[:-1]
            tag_body = " ".join(annot_body)
            # create or get tag ID as relevant
            if tag_body not in mapping_tags:
                mapping_tags[tag_body] = len(mapping_tags)
            tag_id = mapping_tags[tag_body]
            # set up metadata for a new tag (if it already exists, meh)
            if tag_id not in mapping_tagdata:
                mapping_tagdata[tag_id] = {
                    "name": tag_body,
                    "links": {},
                }
            # log interaction info
            work_id = r["id_item"]
            if work_id not in mapping_tagdata[tag_id]["links"]:
                mapping_tagdata[tag_id]["links"][work_id] = {
                    "relevant": True,
                    "score_relevant": None,  # TODO can improve
                    "score_known": None,  # TODO can improve
                    # Add any other custom relevance info here...
                }
    # TODO add anything that can add "False" interaction information?

    # finally, convert mappings to taginfo and taglinks
    dict_taginfo = {
        "id_tag": [],
        "name": [],
    }
    dict_taglinks = {
        "id_item": [],
        "id_tag": [],
        "relevant": [],
        "score_relevant": [],
        "score_known": [],
        # Add any other custom relevance info here...
    }
    for tag_id, tag_info in mapping_tagdata.items():
        dict_taginfo["id_tag"].append(tag_id)
        dict_taginfo["name"].append(tag_info["name"])
        for movie_id, tag_movie_info in tag_info["links"].items():
            dict_taglinks["id_item"].append(movie_id)
            dict_taglinks["id_tag"].append(tag_id)
            dict_taglinks["relevant"].append(tag_movie_info["relevant"])
    df_taginfo = pd.DataFrame(dict_taginfo)
    df_taglinks = pd.DataFrame(dict_taglinks)

    # verify that output matches format minimum
    validate_tag_info_df(df_taginfo)
    validate_tag_links_df(df_taglinks)
    # return results
    return df_taginfo, df_taglinks
