"""Policy-based (per-source) performance metrics for BTSbot predictions.

A policy turns a source's alert-level predictions, accumulated in time order, into a
single "save this source" decision. Used by val.diagnostic_fig and by any model that
produces per-alert scores (e.g. btsbot/tfm_example.ipynb).

Kept free of torch and matplotlib so it can be imported anywhere.
"""
import numpy as np
import pandas as pd


def bts_p1(alerts):
    valid = alerts[(alerts['preds'] == 1) & (alerts['magpsf'] < 19)]
    return len(valid) >= 2


def bts_p2(alerts):
    if np.min(alerts['magpsf']) <= 18.5:
        valid = alerts[(alerts['preds'] == 1) & (alerts['magpsf'] < 19)]
        return len(valid) >= 2
    return False


def prod_p1(alerts):
    valid = alerts[(alerts['raw_preds'] > 0.85) & (alerts['magpsf'] < 19)]
    return len(valid) >= 1


def prod_p2(alerts):
    if np.min(alerts['magpsf']) <= 18.5:
        valid = alerts[(alerts['raw_preds'] > 0.85) & (alerts['magpsf'] < 19)]
        return len(valid) >= 1
    return False


POLICIES = {"bts_p1": bts_p1, "bts_p2": bts_p2, "prod_p1": prod_p1, "prod_p2": prod_p2}


# Cumulative forms of the policies above, used by compute_policy_performance_fast.
# Each takes `a`, a dict of numpy arrays holding every alert, grouped by source and
# time-sorted within each source, plus `gcount(mask)`, which gives the running count of
# True values within each source. It returns the policy's decision after each alert, i.e.
# out[i] == policy(alerts of that source up to and including alert i).
# `np.min(magpsf) <= 18.5` over a history (pandas' np.min skips NaNs) is equivalent to
# "some alert so far has magpsf <= 18.5", which is what the count > 0 terms express.
def bts_p1_cumulative(a, gcount):
    return gcount((a["preds"] == 1) & (a["magpsf"] < 19)) >= 2


def bts_p2_cumulative(a, gcount):
    return (gcount(a["magpsf"] <= 18.5) > 0) & bts_p1_cumulative(a, gcount)


def prod_p1_cumulative(a, gcount):
    return gcount((a["raw_preds"] > 0.85) & (a["magpsf"] < 19)) >= 1


def prod_p2_cumulative(a, gcount):
    return (gcount(a["magpsf"] <= 18.5) > 0) & prod_p1_cumulative(a, gcount)


CUMULATIVE_POLICIES = {"bts_p1": bts_p1_cumulative, "bts_p2": bts_p2_cumulative,
                       "prod_p1": prod_p1_cumulative, "prod_p2": prod_p2_cumulative}


def compute_policy_performance(cand,
                               trues_path="data/base_data/trues.csv",
                               rcf_junk_path="data/base_data/RCFJunk_Feb21_2025.csv"):
    """Run every policy in POLICIES over each source's alerts in time order.

    Args:
        cand: One row per alert, with columns objectId, jd, label, magpsf, peakmag,
            raw_preds (score in [0, 1]) and preds (0/1 prediction). Every alert of a
            source should be present, since policies accumulate over a source's history.
        trues_path: CSV with RCF save/trigger times for BTS sources (ZTFID column).
        rcf_junk_path: CSV of RCF junk sources (id column), excluded from the metrics.

    Returns:
        policy_cand: One row per evaluated source, with each policy's prediction,
            first-save jd/mag and time-to-save / time-to-trigger columns.
        policy_performance: {policy name: {policy_precision, policy_recall,
            binned_precision, binned_recall, peakmag_bins, med_save_dt, med_trigger_dt}}.
            Values are -999.0 when a policy has no true positives or no true negatives.
    """
    cand = cand.reset_index(drop=True)

    save_times = pd.read_csv(trues_path).set_index("ZTFID")['RCF_save_time'].to_dict()
    trigger_times = pd.read_csv(trues_path).set_index("ZTFID")['RCF_trigger_time'].to_dict()
    RCFJunk = pd.read_csv(rcf_junk_path, index_col=None)

    policy_performance = dict.fromkeys(POLICIES)

    # Get label and peakmag for each source (by taking all unique objectIds)
    policy_cand = pd.DataFrame(columns=["objectId", "label", "peakmag",
                                        "remaining_alert_peakmag"])
    # Iterate over all alerts in validation/test set
    for objid in pd.unique(cand['objectId']):
        obj_alerts = cand[cand['objectId'] == objid]

        already_seen = objid in policy_cand['objectId'].to_numpy()
        in_RCFJunk = objid in RCFJunk['id'].to_numpy()
        good_coverage = len(obj_alerts) >= 2  # improve, change to quality cut?
        is_bts = obj_alerts["label"].iloc[0] == 1
        min_magpsf = np.min(obj_alerts["magpsf"])
        BTS_peak_thinned = is_bts and min_magpsf > 18.5

        if (
            (not already_seen) and
            (not in_RCFJunk) and
            (good_coverage) and
            (not BTS_peak_thinned)
        ):
            policy_cand.loc[len(policy_cand)] = (
                objid,
                cand.loc[cand['objectId'] == objid, "label"].iloc[0],
                cand.loc[cand['objectId'] == objid, "peakmag"].iloc[0],
                np.min(cand.loc[cand['objectId'] == objid, "magpsf"])
            )

    # For each policy
    for name, func in POLICIES.items():
        # Initialize new columns
        policy_cand[name + "_pred"] = 0
        policy_cand[name + "_jd"] = np.float64(-1)
        policy_cand[name + "_mag"] = np.float64(-1)
        policy_cand[name + "_save_dt"] = np.nan
        policy_cand[name + "_trigger_dt"] = np.nan

        # For each source
        for obj_id in policy_cand["objectId"]:
            # Pick out alerts for that source and sort them by time
            obj_alerts = cand.loc[cand["objectId"] == obj_id].sort_values(by="jd")

            # For each alert
            for i in range(len(obj_alerts)):
                # the obj_alerts index of the current row of iteration
                idx_cur = obj_alerts.index[i]

                # the obj_alerts index of the current and previous rows of iteration
                idx_sofar = obj_alerts.index[0:i + 1]

                # Don't save before 19 mag
                # if np.min(obj_alerts.loc[obj_alerts.index[0:i+1], 'magpsf']) > 19:
                #     continue

                # Compute the prediction for this current policy
                policy_pred = func(obj_alerts.loc[idx_sofar])

                # If this is the first positive pred
                current_policy_pred_val = policy_cand.loc[
                    policy_cand['objectId'] == obj_id, name + "_pred"
                ].values[0]
                if int(policy_pred) and not current_policy_pred_val:
                    policy_cand.loc[
                        policy_cand['objectId'] == obj_id, name + "_jd"
                    ] = obj_alerts.loc[idx_cur, "jd"]

                    policy_cand.loc[
                        policy_cand['objectId'] == obj_id, name + "_mag"
                    ] = obj_alerts.loc[idx_cur, "magpsf"]

                # Store policy prediction
                policy_cand.loc[
                    policy_cand['objectId'] == obj_id, name + "_pred"
                ] = int(policy_pred)

        policy_labels = policy_cand["label"].to_numpy()
        policy_preds = policy_cand[name + "_pred"].to_numpy()
        bright_narrow_bins = np.arange(17.00, 18.50 + 0.25, 0.25)

        TP_mask_policy = np.bitwise_and(policy_labels, policy_preds)
        TN_mask_policy = 1 - (np.bitwise_or(policy_labels, policy_preds))
        FP_mask_policy = np.bitwise_and(1 - policy_labels, policy_preds)
        FN_mask_policy = np.bitwise_and(policy_labels, 1 - policy_preds)

        TP_idxs_policy = [ii for ii, mi in enumerate(TP_mask_policy) if mi == 1]
        TN_idxs_policy = [ii for ii, mi in enumerate(TN_mask_policy) if mi == 1]
        FP_idxs_policy = [ii for ii, mi in enumerate(FP_mask_policy) if mi == 1]
        FN_idxs_policy = [ii for ii, mi in enumerate(FN_mask_policy) if mi == 1]

        TP_count_policy_binned, _ = np.histogram(
            policy_cand.loc[TP_idxs_policy, "remaining_alert_peakmag"], bins=bright_narrow_bins
        )
        FP_count_policy_binned, _ = np.histogram(
            policy_cand.loc[FP_idxs_policy, "remaining_alert_peakmag"], bins=bright_narrow_bins
        )
        # TN_count_policy_binned, _ = np.histogram(
        #     policy_cand.loc[TN_idxs_policy, "remaining_alert_peakmag"], bins=bright_narrow_bins
        # )
        FN_count_policy_binned, _ = np.histogram(
            policy_cand.loc[FN_idxs_policy, "remaining_alert_peakmag"], bins=bright_narrow_bins
        )

        if all((len(TP_idxs_policy) > 0, len(TN_idxs_policy) > 0)):
            policy_precision = len(TP_idxs_policy) / (len(FP_idxs_policy) + len(TP_idxs_policy))
            policy_recall = len(TP_idxs_policy) / (len(FN_idxs_policy) + len(TP_idxs_policy))

            binned_precision = TP_count_policy_binned / (
                TP_count_policy_binned + FP_count_policy_binned
            )
            binned_recall = TP_count_policy_binned / (
                TP_count_policy_binned + FN_count_policy_binned
            )

            # policy cand has only val for all sets
            # save_times has train+val+test but only BTSSE trues
            # MS trues have unreliable/unrealistic save times, so exclude.
            # Pre ~2021 sources have unreliable Fritz save times (scanned on GROWTH), exclude.
            jan1_2021_jd = 2459215.5
            for objid_tp in policy_cand.loc[TP_idxs_policy, "objectId"].to_list():
                policy_obj_jd_series = policy_cand.loc[
                    policy_cand["objectId"] == objid_tp, name + "_jd"
                ]
                policy_obj_jd = policy_obj_jd_series.values[0]

                if objid_tp in list(save_times):
                    # Some BTS trues don't have save times...
                    if (save_times[objid_tp] >= jan1_2021_jd) and (policy_obj_jd > 0):
                        current_jd = policy_cand.loc[
                            policy_cand["objectId"] == objid_tp, name + "_jd"
                        ].values[0]
                        policy_cand.loc[
                            policy_cand["objectId"] == objid_tp, name + "_save_dt"
                        ] = current_jd - save_times[objid_tp]

                if objid_tp in list(trigger_times):
                    trigger_time_val = trigger_times[objid_tp]
                    if (
                        trigger_time_val >= jan1_2021_jd and
                        trigger_time_val < 1e10 and
                        policy_obj_jd > 0
                    ):
                        current_jd = policy_cand.loc[
                            policy_cand["objectId"] == objid_tp, name + "_jd"
                        ].values[0]
                        policy_cand.loc[
                            policy_cand["objectId"] == objid_tp, name + "_trigger_dt"
                        ] = current_jd - trigger_time_val

            med_save_dt = np.nanmedian(policy_cand[name + "_save_dt"])
            med_trigger_dt = np.nanmedian(policy_cand[name + "_trigger_dt"])
        else:
            policy_precision = -999.0
            policy_recall = -999.0
            binned_precision = [-999.0]
            binned_recall = [-999.0]
            med_save_dt = -999.0
            med_trigger_dt = -999.0

        policy_performance[name] = {
            "policy_precision": policy_precision,
            "policy_recall": policy_recall,
            "binned_precision": list(binned_precision),
            "binned_recall": list(binned_recall),
            "peakmag_bins": list(bright_narrow_bins),
            "med_save_dt": med_save_dt,
            "med_trigger_dt": med_trigger_dt
        }

        print(f"Finished policy {name} analysis")

    return policy_cand, policy_performance


def _argsort_like_pandas(values):
    """The row order DataFrame.sort_values(by=<this column>) produces (default quicksort,
    NaNs last). Mirrored exactly so alerts with tied jd come out in the same order."""
    nan = pd.isna(values)
    if not nan.any():
        return values.argsort(kind="quicksort")
    idx = np.flatnonzero(~nan)
    return np.concatenate([idx[values[idx].argsort(kind="quicksort")], np.flatnonzero(nan)])


def compute_policy_performance_fast(cand,
                                    trues_path="data/base_data/trues.csv",
                                    rcf_junk_path="data/base_data/RCFJunk_Feb21_2025.csv"):
    """Vectorized equivalent of compute_policy_performance, with identical outputs.

    Instead of re-running each policy on every alert's history, every policy is evaluated
    for all sources and alerts at once with the cumulative forms in CUMULATIVE_POLICIES.
    Same arguments and return values as compute_policy_performance.
    """
    cand = cand.reset_index(drop=True)

    trues = pd.read_csv(trues_path).set_index("ZTFID")
    save_times = trues['RCF_save_time'].to_dict()
    trigger_times = trues['RCF_trigger_time'].to_dict()
    junk_ids = set(pd.read_csv(rcf_junk_path, index_col=None)['id'])

    # Group alerts by source, sources in order of first appearance (as pd.unique), alerts
    # in their original order within each source. Rows with a missing objectId are never
    # evaluated by the original either (they match no source).
    codes, _ = pd.factorize(cand["objectId"])
    order = np.argsort(codes, kind="stable")
    order = order[codes[order] >= 0]
    counts = np.bincount(codes[codes >= 0])
    starts = np.cumsum(counts) - counts
    first_row = order[starts]  # each source's first alert in the original order

    # Which sources are evaluated (same cuts as compute_policy_performance)
    objids = cand["objectId"].to_numpy()[first_row]
    labels = cand["label"].to_numpy()[first_row]
    magpsf_grouped = cand["magpsf"].to_numpy()[order]
    min_magpsf = np.fmin.reduceat(magpsf_grouped, starts) if len(starts) else magpsf_grouped
    in_junk = np.array([oid in junk_ids for oid in objids], dtype=bool)
    peak_thinned = (labels == 1) & (min_magpsf > 18.5)
    evaluated = ~in_junk & (counts >= 2) & ~peak_thinned

    if evaluated.any():
        policy_cand = pd.DataFrame({
            "objectId": objids[evaluated],
            "label": labels[evaluated],
            "peakmag": cand["peakmag"].to_numpy()[first_row[evaluated]],
            "remaining_alert_peakmag": min_magpsf[evaluated],
        })
    else:
        policy_cand = pd.DataFrame(columns=["objectId", "label", "peakmag",
                                            "remaining_alert_peakmag"])

    # Alerts of the evaluated sources, grouped by source and sorted by jd within each
    jd_all = cand["jd"].to_numpy()
    seg_rows = []
    for g in np.flatnonzero(evaluated):
        rows = order[starts[g]:starts[g] + counts[g]]
        seg_rows.append(rows[_argsort_like_pandas(jd_all[rows])])
    rows = np.concatenate(seg_rows) if seg_rows else np.zeros(0, dtype=np.int64)
    alerts = {col: cand[col].to_numpy()[rows] for col in ["jd", "magpsf", "raw_preds", "preds"]}

    n_alerts = len(rows)
    g_counts = counts[evaluated]
    g_starts = np.cumsum(g_counts) - g_counts
    g_ends = g_starts + g_counts
    source_of_alert = np.repeat(np.arange(len(g_counts)), g_counts)

    def gcount(mask):
        mask = np.asarray(mask, dtype=np.int64)
        total = np.cumsum(mask)
        before_source = total[g_starts] - mask[g_starts]
        return total - before_source[source_of_alert]

    policy_performance = dict.fromkeys(POLICIES)
    for name in POLICIES:
        cumulative_pred = np.asarray(CUMULATIVE_POLICIES[name](alerts, gcount), dtype=bool)

        # The original records jd/mag whenever the policy flips from 0 (its value after the
        # previous alert, initially 0) to 1, so the stored save is the last such flip, and
        # the stored prediction is the policy's value after the source's final alert.
        prev_pred = np.zeros(n_alerts, dtype=bool)
        prev_pred[1:] = cumulative_pred[:-1]
        prev_pred[g_starts] = False
        flips = cumulative_pred & ~prev_pred
        last_flip = (np.maximum.reduceat(np.where(flips, np.arange(n_alerts), -1), g_starts)
                     if n_alerts else np.zeros(0, dtype=np.int64))
        has_flip = last_flip >= 0

        save_jd = np.full(len(g_counts), np.float64(-1))
        save_mag = np.full(len(g_counts), np.float64(-1))
        save_jd[has_flip] = alerts["jd"][last_flip[has_flip]]
        save_mag[has_flip] = alerts["magpsf"][last_flip[has_flip]]

        policy_cand[name + "_pred"] = cumulative_pred[g_ends - 1].astype(np.int64)
        policy_cand[name + "_jd"] = save_jd
        policy_cand[name + "_mag"] = save_mag
        policy_cand[name + "_save_dt"] = np.nan
        policy_cand[name + "_trigger_dt"] = np.nan

        policy_performance[name] = _policy_summary(policy_cand, name, save_times, trigger_times)
        print(f"Finished policy {name} analysis")

    return policy_cand, policy_performance


def _policy_summary(policy_cand, name, save_times, trigger_times):
    """Precision/recall and time-to-save for one policy, as in compute_policy_performance.
    Also fills policy_cand's <name>_save_dt and <name>_trigger_dt columns."""
    policy_labels = policy_cand["label"].to_numpy()
    policy_preds = policy_cand[name + "_pred"].to_numpy()
    bright_narrow_bins = np.arange(17.00, 18.50 + 0.25, 0.25)

    TP_mask_policy = np.bitwise_and(policy_labels, policy_preds)
    TN_mask_policy = 1 - (np.bitwise_or(policy_labels, policy_preds))
    FP_mask_policy = np.bitwise_and(1 - policy_labels, policy_preds)
    FN_mask_policy = np.bitwise_and(policy_labels, 1 - policy_preds)

    TP_idxs_policy = [ii for ii, mi in enumerate(TP_mask_policy) if mi == 1]
    TN_idxs_policy = [ii for ii, mi in enumerate(TN_mask_policy) if mi == 1]
    FP_idxs_policy = [ii for ii, mi in enumerate(FP_mask_policy) if mi == 1]
    FN_idxs_policy = [ii for ii, mi in enumerate(FN_mask_policy) if mi == 1]

    TP_count_policy_binned, _ = np.histogram(
        policy_cand.loc[TP_idxs_policy, "remaining_alert_peakmag"], bins=bright_narrow_bins
    )
    FP_count_policy_binned, _ = np.histogram(
        policy_cand.loc[FP_idxs_policy, "remaining_alert_peakmag"], bins=bright_narrow_bins
    )
    FN_count_policy_binned, _ = np.histogram(
        policy_cand.loc[FN_idxs_policy, "remaining_alert_peakmag"], bins=bright_narrow_bins
    )

    if all((len(TP_idxs_policy) > 0, len(TN_idxs_policy) > 0)):
        policy_precision = len(TP_idxs_policy) / (len(FP_idxs_policy) + len(TP_idxs_policy))
        policy_recall = len(TP_idxs_policy) / (len(FN_idxs_policy) + len(TP_idxs_policy))

        binned_precision = TP_count_policy_binned / (
            TP_count_policy_binned + FP_count_policy_binned
        )
        binned_recall = TP_count_policy_binned / (
            TP_count_policy_binned + FN_count_policy_binned
        )

        # Time from the policy's save to the scanners' save / trigger, for true positives
        # with reliable (2021+) times; see compute_policy_performance for the reasoning.
        jan1_2021_jd = 2459215.5
        is_tp = np.zeros(len(policy_cand), dtype=bool)
        is_tp[TP_idxs_policy] = True
        policy_jd = policy_cand[name + "_jd"].to_numpy()
        save_time = policy_cand["objectId"].map(save_times).to_numpy(dtype=np.float64)
        trigger_time = policy_cand["objectId"].map(trigger_times).to_numpy(dtype=np.float64)

        has_save = is_tp & (save_time >= jan1_2021_jd) & (policy_jd > 0)
        has_trigger = (is_tp & (trigger_time >= jan1_2021_jd) & (trigger_time < 1e10) &
                       (policy_jd > 0))
        save_dt = np.full(len(policy_cand), np.nan)
        trigger_dt = np.full(len(policy_cand), np.nan)
        save_dt[has_save] = policy_jd[has_save] - save_time[has_save]
        trigger_dt[has_trigger] = policy_jd[has_trigger] - trigger_time[has_trigger]
        policy_cand[name + "_save_dt"] = save_dt
        policy_cand[name + "_trigger_dt"] = trigger_dt

        med_save_dt = np.nanmedian(policy_cand[name + "_save_dt"])
        med_trigger_dt = np.nanmedian(policy_cand[name + "_trigger_dt"])
    else:
        policy_precision = -999.0
        policy_recall = -999.0
        binned_precision = [-999.0]
        binned_recall = [-999.0]
        med_save_dt = -999.0
        med_trigger_dt = -999.0

    return {
        "policy_precision": policy_precision,
        "policy_recall": policy_recall,
        "binned_precision": list(binned_precision),
        "binned_recall": list(binned_recall),
        "peakmag_bins": list(bright_narrow_bins),
        "med_save_dt": med_save_dt,
        "med_trigger_dt": med_trigger_dt
    }
