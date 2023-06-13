#!/usr/bin/env python3
import requests, urllib, numpy as np, json, time, sys, pandas as pd
import astropy.time as astrotime
import astropy.units as u
BOLD = "\033[1m"; END  = "\033[0m"

host = "https://fritz.science"
metadata_endpoint = "alerts"
triplets_endpoint = "alerts_triplets"

with open('/Users/nabeelr/credentials.json', 'r') as f:
    creds = json.load(f)
    api_token = creds['fritz_api_key']
headers = {'Authorization': f'token {api_token}'}

RCF_groupid = "41"
RCFJunk_groupid = "255"
BTSbot_groupid = "1534"

gt1_groupid = "1555"
gt2_groupid = "1556"
gt3_groupid = "1557"


def gt_N(scores, N):
    return np.sum(np.asarray(scores) > 0.5) >= N


def save_to_group(objid, groupid, group_name=""):
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
        if "Source already saved" not in r.json()['message']:
            print(r.text)
            exit(0)
        else:
            print(f"  already saved with {group_name}")
    else:
        print(f"  {BOLD}saved to {group_name}{END}")


def save_and_summarize(date):
    """
    Save RCF candidates for a specified night to BTS bot test groups if they 
    pass set criteria and create a CSV showing which candidates were saved
    to RCF or BTS bot groups that day 
    
    Parameters
    ----------
    date: string
        Beginning date of 24 hour window to search for candidates from
        Provide string formatted as YYYY-MM-DD
    
    Returns
    -------
    Nothing
    """
    
    # Noon of given date to next day noon (PDT)
    start_date = astrotime.Time(date+"T19:00:00")
    end_date = start_date + 1*u.day

    # Get objectIds of RCF candidates from specified night 
    # Only select those not already saved to RCFJunk
    url = urllib.parse.urljoin(host, f'/api/candidates')
    params = {
        "savedStatus": "notSavedToAnySelected",
        "startDate": start_date.value,
        "endDate": end_date.value,
        "groupIDs": BTSbot_groupid + "," + RCFJunk_groupid,
        "numPerPage": 500,
    }
    r = requests.get(url, headers=headers, params=params)

    objids = [cand['id'] for cand in r.json()['data']['candidates']]
    candidates = pd.DataFrame(objids, columns=["objectId"])
    print(f"Found {len(objids)} RCF candidates")

    # Criteria to run BTS scores through
    critera_names = ["gt1", "gt2", "gt3"]
    critera = [lambda x: gt_N(x, 1), lambda x: gt_N(x, 2), lambda x: gt_N(x, 3)]
    groupids = [gt1_groupid, gt2_groupid, gt3_groupid]

    # Initialize columns to store jd of when each criterion was passed
    for crit_name in critera_names:
        candidates[crit_name+"_savetime"] = None

    candidates['RCF_savetime'] = None

    # Only candidates that were saved 'tonight' (by model using any criterion or RCF scanners)
    # tonight: between 4 PM PDT of selected date to 4 PM PDT of next day
    saved_tonight = pd.DataFrame(columns=candidates.columns)
    
    # For every candidate,
    for objid in objids:
        print(objid)

        # Query fritz for bts scores and their jds
        url = urllib.parse.urljoin(host, f"/api/alerts/{objid}")
        params = {
            "projection": json.dumps({
                "_id": 0,
                "candid": 1,
                "candidate.jd": 1,
                "classifications.bts": 1,
            }),
        }
        r = requests.get(url, headers=headers, params=params)
        if not r.ok:
            print(r.json())
            exit(0)

        dat = r.json()['data']
        # Only keep alerts that have a BTS score
        alerts = [[alr['candid'], alr['candidate']['jd'], alr['classifications']['bts']] for alr in dat if "bts" in list(alr['classifications'])]
        # Store in dataframe and sort by jd
        alerts = pd.DataFrame(alerts, columns=['candid', 'jd', 'bts']).sort_values(by="jd")
        print(f"  found {len(alerts)} alerts with bts scores")
        
        for criterion, groupid, crit_name in zip(critera, groupids, critera_names):
            for i in range(len(alerts)):
                # the alerts index of the current row of iteration
                idx_cur = alerts.index[i]

                # the alerts index of the current and previous rows of iteration
                idx_sofar = alerts.index[0:i+1]

                # Compute the prediction for the current criterion
                crit_pred = criterion(alerts.loc[idx_sofar, 'bts'].to_list())

                # If the source passed criterion, 
                if crit_pred:
                    # Save it to the respective group
                    save_to_group(objid, groupid, crit_name)

                    # If it was the first time passing this criterion, store jd of save
                    if not candidates.loc[candidates['objectId'] == objid, crit_name+"_savetime"].values[0]:
                        candidates.loc[candidates['objectId'] == objid, crit_name+"_savetime"] = alerts.loc[idx_cur, 'jd']
                    
                    # Don't need to check for future alerts because source already passed this criterion
                    break

        # Query for RCF save time
        url = urllib.parse.urljoin(host, f"/api/sources/{objid}")
        r = requests.get(url, headers=headers, params={})

        for group in r.json()['data']['groups']:
            if group['name'] == "Redshift Completeness Factor":
                print("  saved to RCF at", astrotime.Time(group['saved_at']).jd)
                candidates.loc[candidates['objectId'] == objid, "RCF_savetime"] = astrotime.Time(group['saved_at']).jd
            
        # jd of when every criterion and scanners saved source 
        save_times = candidates[candidates['objectId'] == objid].to_numpy()[0,1:]

        for save_time in save_times:
            # check if any saving happened in the specified time window
            if save_time and (save_time - start_date.jd > 0) and (save_time - end_date.jd < 0):
                print(f'  {BOLD}saved tonight{END}')
                saved_tonight = pd.concat((saved_tonight, candidates[candidates['objectId'] == objid]))
                break

        # too many requests too quickly gets you rate limited
        time.sleep(0.1)

    saved_tonight.to_csv(f"nightly_summaries/{start_date.strftime('%Y-%m-%d')}.csv", index=None)
    print(f"Done autoscan for {start_date.strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    try:
        date = astrotime.Time(sys.argv[1]).strftime('%Y-%m-%d')
    except IndexError:
        date = (astrotime.Time.now() - 1*u.day).strftime('%Y-%m-%d')

    save_and_summarize(date)