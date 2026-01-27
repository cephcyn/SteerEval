from pathlib import Path
import pprint

import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm

focus_dir = "output/job_dir"
do_merge_all_csv = False
colname_job_name = "INFO:job_name"
colnames_special = [
    "INFO:history_size",
    "INFO:pool_next",
    "INFO:pool_related",
    "INFO:pool_unrelated",
    "INFO:data_tag",
    "INFO:steer_action",
    "SAMPLE:selected_tag",
    "INFO:profile_method",
    "INFO:profile_data",
    "INFO:steering_method",
    "INFO:ranking_method",
    "INFO:ranking_data",
    "PROFILE:profile_original",
    "STEERING:profile_steered",
]


def should_remove_from_merge(
    s: str,
) -> bool:
    should_be_removed = False
    if "REF" in s:
        should_be_removed = True
    return should_be_removed


def colname_is_metric(
    c: str,
) -> bool:
    is_metric = False
    if ("@" in c) or ("minrank" in c):
        is_metric = True
    if c.split(":")[0] == "SCORES":
        is_metric = True
    return is_metric


def colname_metric_delta(
    c: str,
) -> tuple[bool, str, str]:
    is_delta = False
    c_split = c.split(":")
    if len(c_split) >= 2 and c_split[1] == "DELTA":
        is_delta = True
    if not c_split:
        return False, "", ""
    else:
        c_orign = ":".join([c_split[0], "ORIGN"] + c_split[2:])
        c_steer = ":".join([c_split[0], "STEER"] + c_split[2:])
        return is_delta, c_orign, c_steer


output_dir = focus_dir

focus_dir_path = Path(focus_dir)
output_dir_path = Path(output_dir)

job_dirs = [f for f in focus_dir_path.iterdir() if f.is_dir()]

# merged results
results_all = {}
# scan through job results
print("Doing initial scanning")
for j in tqdm(job_dirs):
    if str(j.name)[0] == ".":
        continue
    # load results csv
    results_fname = j / "results.csv"
    if results_fname.is_file():
        df_results_single = pd.read_csv(results_fname)
        # add a column identifying the name of the dir-within (the job name)
        df_results_single.insert(0, "job_name", j.name)
        # df_results_single["job_name"] = j.name
        if do_merge_all_csv:
            # append everything to the big group
            results_all[j.name] = df_results_single
        else:
            # we are just logging col names for now, this is irrelevant data
            results_all[j.name] = df_results_single.iloc[[0]].copy()
    else:
        results_all[j.name] = None
# create big merged results
df_results = pd.concat(results_all.values(), ignore_index=True)
# with pd.option_context('display.max_rows', 200):
#     df_results
#     # pprint.pprint(df_results)
if do_merge_all_csv:
    # individual small
    for j_name, df_r in results_all.items():
        if df_r is not None:
            single_path = Path(output_dir_path / "SINGLE").mkdir(
                parents=True, exist_ok=True
            )
            df_r.to_csv(single_path / f"SINGLE_{j_name}.csv", index=False)
        else:
            print(f"Incomplete or 0 sampled: {j_name}")
    # large merged
    df_results.to_csv(output_dir_path / "MERGED.csv", index=False)
    # remove reference random sampling rows from merged results...
    for j in set(df_results["job_name"]):
        if should_remove_from_merge(j):
            df_results = df_results[df_results["job_name"] != j]
    df_results.to_csv(output_dir_path / "MERGED_short.csv", index=False)

# create a sample-table with example prompts and profiles
# AND
# create an aggregate-table with average evaluation numbers across all jobs
aggregated = {
    "job_name": [],
    "sample_size": [],
}

# get metric names
metric_names = [c for c in df_results.columns if colname_is_metric(c)]

# scan through job results a second time
print("Doing aggregation calculation")
skipped_dirs = []
for j in tqdm(job_dirs):
    if str(j.name)[0] == ".":
        skipped_dirs.append(j.name)
        continue
    # load results csv
    results_fname = j / "results.csv"
    if results_fname.is_file():
        df_results_single = pd.read_csv(results_fname)
        # add a column identifying the name of the dir-within (the job name)
        df_results_single.insert(0, "job_name", j.name)
        # now calculate aggregated stuff
        aggregated["job_name"].append(j.name)
        # sampling things
        single_example = df_results_single.iloc[0]
        for c in colnames_special:
            if c not in aggregated:
                aggregated[c] = []
            aggregated[c].append(single_example[c])
        # aggregating metrics
        aggregated["sample_size"].append(len(df_results_single))
        for m in metric_names:
            if m not in aggregated:
                aggregated[m] = []
            aggregated[m].append(df_results_single[m].mean())
            # attempt conf interval, ttests
            m_is_metric, m_orign, m_steer = colname_metric_delta(m)
            if m_is_metric:
                metric_ci = stats.t.interval(
                    0.95,
                    len(df_results_single[m]) - 1,
                    loc=np.mean(df_results_single[m]),
                    scale=stats.sem(df_results_single[m]),
                )
                agg_ci_name = f"ci_{m}"
                if agg_ci_name not in aggregated:
                    aggregated[agg_ci_name] = []
                aggregated[agg_ci_name].append(metric_ci)
                metric_ttest = stats.ttest_rel(
                    df_results_single[m_orign],
                    df_results_single[m_steer],
                )
                agg_tt_name = f"ttest_{m}"
                if agg_tt_name not in aggregated:
                    aggregated[agg_tt_name] = []
                aggregated[agg_tt_name].append(metric_ttest)
    else:
        skipped_dirs.append(j.name)

# aggregate
df_aggregated = pd.DataFrame(aggregated)
print(f"df shape {df_aggregated.shape}")
output_fname = output_dir_path / "AGGREGATE.csv"
df_aggregated.to_csv(output_fname, index=False)
# print aggregate size
print(f"written to {output_fname}")
print(f"{len(df_aggregated)} aggregate out of {len(job_dirs)} job dirs")
print(f"skipped: {skipped_dirs}")

# with pd.option_context(
#     "display.max_rows", None,
#     "display.max_columns", None,
#     "display.max_colwidth", None,
# ):
#     df_aggregated
#     # pprint.pprint(df_aggregated)
