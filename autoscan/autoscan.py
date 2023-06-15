#!/usr/bin/env python3
import requests, urllib, numpy as np, json, time, os, sys, pandas as pd
import astropy.time as astrotime
import astropy.units as u
BOLD = "\033[1m"; END  = "\033[0m"

host = "https://fritz.science"
metadata_endpoint = "alerts"
triplets_endpoint = "alerts_triplets"

if sys.platform == "darwin":
    base_path = "/Users/nabeelr/Desktop/School/ZTF Research/BNB-classifier/"
    creds_path = "/Users/nabeelr/credentials.json"
else:
    base_path = "/projects/b1094/rehemtulla/BNB-classifier/"
    creds_path = f"{base_path}misc/credentials.json"

with open(creds_path, 'r') as f:
    creds = json.load(f)
    api_token = creds['fritz_api_key']
headers = {'Authorization': f'token {api_token}'}

RCF_groupid = "41"
RCFJunk_groupid = "255"
BTSbot_groupid = "1534"

gt1_groupid = "1583"
gt2_groupid = "1584"
gt3_groupid = "1585"


def gt_N(scores, N):
    return np.sum(np.asarray(scores) > 0.5) >= N


def save_to_group(objid, groupid, group_name="", retry=False):
    """
    Save source with objid to fritz group with groupid and name group_name
    
    Parameters
    ----------
    objid: string
        ZTF objectId of source to save

    groupid: string
        ID of Fritz group to save source to

    group_name (optional): string
        name of Fritz group - used for printed status updates  
    
    Returns
    -------
    Nothing
    """

    url = urllib.parse.urljoin(host, f"/api/source_groups")
    params = {
        "objId": objid,
        "inviteGroupIds": [groupid]
    }
    r = requests.post(url, headers=headers, json=params)

    if not r.ok:
        try:
            message = r.json()['message']
            # print(r.text)
        except Exception as e:
            print("failed to extract message from response")
            print(e)
            message = ""

        if "Source already saved" in message:
            print(f"  already saved with {group_name}")
        elif not retry:
            print(f"retrying save of {objid} to {group_name}={groupid}")
            time.sleep(1)
            save_to_group(objid, groupid, group_name, retry=True)
            return
        else:
            print(f"save of {objid} to {group_name}={groupid} failed on retry")
    else:
        print(f"  {BOLD}saved to {group_name}{END}")


def get_candidates(start_date, end_date):
    """
    Get RCF candidates between start_ and end_date that are not RCF_Junk

    Parameters
    ----------
    start_date and end_date
    astropy time objects indicating UTC time window of search
    
    Returns
    -------
    candidates
    DataFrame with column "objectId" containing ZTF objectIds for sources that 
    had alerts pass the RCF filter in the start_ end_date window
    """
    

    # Get objectIds of RCF candidates from specified night 
    # Only select those not already saved to RCFJunk
    url = urllib.parse.urljoin(host, f'/api/candidates')
    params = {
        "savedStatus": "notSavedToAnySelected",
        "startDate": start_date.value,  # expects UTC
        "endDate": end_date.value,
        "groupIDs": BTSbot_groupid + "," + RCFJunk_groupid,
        "numPerPage": 500,
    }
    r = requests.get(url, headers=headers, params=params)

    try:
        objids = [cand['id'] for cand in r.json()['data']['candidates']]
    except Exception as e:
        print(e)
        print("failed to get candidates")
        objids = []

    candidates = pd.DataFrame(objids, columns=["objectId"])
    print(f"Found {len(objids)} RCF candidates")

    return candidates


def save_log(start_date, end_date, saved_now):
    """
    Write log of sources saved in time window to disk as a CSV
    Naming is slightly different based on the size of the time window

    Parameters
    ----------
    start_date and end_date
    astropy time objects indicating UTC time window of search
    
    saved_now
    DataFrame with columns objectId, and save times of each policy and RCF scanners

    Returns
    -------
    Nothing
    """
    night_path = f"{base_path}autoscan/nightly_summaries/{end_date.strftime('%h%d')}/"
    if not os.path.exists(night_path):
        os.makedirs(night_path)

    if end_date - start_date < 1*u.hr:
        filename = f"{(start_date+5*u.min).strftime('%H%M')}.csv"
    else:
        filename = f"since{start_date.strftime('%h%d_%H%M')}.csv"

    saved_now.to_csv(f"{night_path}{filename}", index=None)
    

def autoscan(start_date, end_date):
    """
    Save RCF candidates to BTS policy groups they pass and create a CSV showing 
    which candidates were saved to RCF or BTSbot groups in past 30 minutes

    Intended to be run every 30 minutes from 8PM-10AM CT with cron:
        */30 20-23,0-10 * * * path/python3 path/autoscan_30min.py >> path/log.log 2>&1

    Times are in UTC unless otherwise specified 
    
    Parameters
    ----------
    start_date and end_date
    astropy time objects indicating UTC time window of search
    
    Returns
    -------
    Nothing
    """
    
    # New RCF candidates from specified time window
    candidates = get_candidates(start_date, end_date)

    # Policy to map BTS alert-based scores to source-based classification
    policy_names = ["gt1", "gt2", "gt3"]
    policies = [lambda x: gt_N(x, 1), lambda x: gt_N(x, 2), lambda x: gt_N(x, 3)]
    groupids = [gt1_groupid, gt2_groupid, gt3_groupid]

    # Initialize columns to store jd of when each policy was passed
    for pol_name in policy_names:
        candidates[pol_name+"_savetime"] = None
    candidates['RCF_savetime'] = None

    # Only candidates that were saved in the past 35 mins 
    # by model using any policy or by RCF scanners
    saved_now = pd.DataFrame(columns=candidates.columns)
    
    # For every candidate,
    for objid in candidates['objectId'].to_numpy():
        print(objid)

        # Query fritz for bts scores and their jds
        url = urllib.parse.urljoin(host, f"/api/alerts/{objid}")
        params = {
            "projection": json.dumps({
                "_id": 0,
                "candid": 1,
                "candidate.jd": 1,
                "classifications.bts": 1,
                "classifications.bts_version": 1,
            }),
        }
        try:
            r = requests.get(url, headers=headers, params=params)
        except Exception as e:
            print(e)
            print(r, url, params)
            print("Failed to query alerts of {objid}\nSkipping to next source")
            continue
        if not r.ok:
            print(r.text)
            print("Failed to unpack alerts of {objid}\nSkipping to next source")
            print(f"Skipping {objid}")
            continue

        dat = r.json()['data']
        # Only keep alerts that have a BTS score from model version v03
        alerts = []
        for alert in dat:
            if "bts" in list(alert['classifications']) and alert['classifications']['bts_version'] == "v03":
                alerts += [[alert['candid'], 
                            alert['candidate']['jd'], 
                            alert['classifications']['bts']]]

        # Store in dataframe and sort by jd
        alerts = pd.DataFrame(alerts, columns=['candid', 'jd', 'bts']).sort_values(by="jd")
        print(f"  found {len(alerts)} alerts with bts scores")
        
        for policy, groupid, pol_name in zip(policies, groupids, policy_names):
            for i in range(len(alerts)):
                # the alerts index of the current row of iteration
                idx_cur = alerts.index[i]

                # the alerts index of the current and previous rows of iteration
                idx_sofar = alerts.index[0:i+1]

                # Compute the prediction for the current policy
                pol_pred = policy(alerts.loc[idx_sofar, 'bts'].to_list())

                # If the source passed policy, 
                if pol_pred:
                    # Save it to the respective group
                    save_to_group(objid, groupid, pol_name)

                    # If it was the first time passing this policy, store jd of save
                    if not candidates.loc[candidates['objectId'] == objid, pol_name+"_savetime"].values[0]:
                        candidates.loc[candidates['objectId'] == objid, pol_name+"_savetime"] = alerts.loc[idx_cur, 'jd']
                    
                    # Don't need to check for future alerts because source already passed this policy
                    break

        # Query for RCF save time
        url = urllib.parse.urljoin(host, f"/api/sources/{objid}")
        r = requests.get(url, headers=headers, params={})

        for group in r.json()['data']['groups']:
            if group['name'] == "Redshift Completeness Factor":
                print("  saved to RCF at", astrotime.Time(group['saved_at']).jd)
                candidates.loc[candidates['objectId'] == objid, "RCF_savetime"] = astrotime.Time(group['saved_at']).jd
            
        # jd of when every policy and scanners saved source 
        # array([gt1_savetime, gt2_savetime, gt3_savetime, RCF_savetime])
        save_times = candidates[candidates['objectId'] == objid].to_numpy()[0,1:]

        for save_time in save_times:
            # check if any saving happened in the specified time window
            if save_time and (save_time - start_date.jd > 0) and (save_time - end_date.jd < 0):
                print(f'  {BOLD}saved now{END}')
                saved_now = pd.concat((saved_now, candidates[candidates['objectId'] == objid]))
                break

        # too many requests too quickly gets you rate limited
        time.sleep(1.0)

    save_log(start_date, end_date, saved_now)
    print(f"Done autoscan for {start_date.strftime('%h%d %H:%M')} to {end_date.strftime('%h%d %H:%M')}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Must provide scan type")
        exit(1)
    scan_type = sys.argv[1]
    
    if scan_type == "during_night":
        # Select candidates from 35 minutes ago to now 
        end_date = astrotime.Time.now()  # in UTC
        start_date = end_date - 35*u.min
    elif scan_type == "after_night":
        # Select candidates from the past 24 hours
        end_date = astrotime.Time.now()  # in UTC
        start_date = end_date - 24*u.hr
    else:
        print(f"could not understand scan_type {scan_type}")
        print("defaulting to 'after_night'")
        end_date = astrotime.Time.now()  # in UTC
        start_date = end_date - 24*u.hr
        
    autoscan(start_date, end_date)
