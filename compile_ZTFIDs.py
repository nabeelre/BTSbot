import numpy as np
import pandas as pd
from tqdm import tqdm
import json, requests, os, time, sys
from astropy import time as astrotime

btsse_query_urls = {
    "trues": "http://sites.astro.caltech.edu/ztf/rcf/explorer.php?f=s&coverage=any&samprcf=y&sampdeep=y&subsample=trans&classstring=&classexclude=&refok=y&purity=y&ps1img=y&lcfig=y&ztflink=fritz&startsavedate=&startpeakdate=&startlastdate=&startra=&startdec=&startz=&startdur=&startrise=&startfade=&startpeakmag=&startlastmag=&startabsmag=&starthostabs=&starthostcol=&startsavevis=&startlatevis=&startcurrvis=&startb=&startav=&endsavedate=&endpeakdate=&endlastdate=&endra=&enddec=&endz=&enddur=&endrise=&endfade=&endpeakmag=18.5&endlastmag=&endabsmag=&endhostabs=&endhostcol=&endsavevis=&endlatevis=&endcurrvis=&endb=&endav=&sort=peakmag&format=csv",
    "vars":  "http://sites.astro.caltech.edu/ztf/rcf/explorer.php?f=s&coverage=any&samprcf=y&sampdeep=y&subsample=var&classstring=&classexclude=&refok=y&lcfig=y&ztflink=fritz&startsavedate=&startpeakdate=&startlastdate=&startra=&startdec=&startz=&startdur=&startrise=&startfade=&startpeakmag=&startlastmag=&startabsmag=&starthostabs=&starthostcol=&startsavevis=&startlatevis=&startcurrvis=&startb=&startav=&endsavedate=&endpeakdate=&endlastdate=&endra=&enddec=&endz=&enddur=&endrise=&endfade=&endpeakmag=&endlastmag=&endabsmag=&endhostabs=&endhostcol=&endsavevis=&endlatevis=&endcurrvis=&endb=&endav=&sort=peakmag&format=csv",
    "dims":  "http://sites.astro.caltech.edu/ztf/rcf/explorer.php?f=s&coverage=any&samprcf=y&sampdeep=y&subsample=all&classstring=&classexclude=&covok=y&refok=y&purity=y&lcfig=y&ztflink=fritz&startsavedate=&startpeakdate=&startlastdate=&startra=&startdec=&startz=&startdur=&startrise=&startfade=&startpeakmag=18.5&startlastmag=&startabsmag=&starthostabs=&starthostcol=&startsavevis=&startlatevis=&startcurrvis=&startb=&startav=&endsavedate=&endpeakdate=&endlastdate=&endra=&enddec=&endz=&enddur=&endrise=&endfade=&endpeakmag=&endlastmag=&endabsmag=&endhostabs=&endhostcol=&endsavevis=&endlatevis=&endcurrvis=&endb=&endav=&sort=peakmag&format=csv",
}

if sys.platform == "darwin":
    with open('/Users/nabeelr/credentials.json', 'r') as f:
        creds = json.load(f)
else:
    with open('misc/credentials.json', 'r') as f:
        creds = json.load(f)

api_token = creds['fritz_api_key']
host = "https://fritz.science"
headers = {'Authorization': f'token {api_token}'}


def query_rejects():
    """
    Query BTS candidates from 2021-2023 that were not saved to RCF
    and are not in RCFjunk

    Parameters
    ----------
    None

    Returns
    -------
    Nothing
    """
    print("Querying for BTS rejects")
    # BTS filter last changed 2020-10-29 
    # Round start_date to 2021-01-01
    start_date = "2021-01-01"
    end_date = "2023-01-01"

    RCF_groupid = "41"
    RCFJunk_groupid = "255"

    endpoint = host+"/api/candidates"

    objids = []
    page_num = 1
    num_per_page = 250

    while True:
        print(f"  Page {page_num} of rejects queries")
        found_new = False
        
        params = {
            "savedStatus": "notSavedToAnySelected",
            "startDate": start_date,
            "endDate": end_date,
            "groupIDs": RCF_groupid + "," + RCFJunk_groupid,
            "numPerPage": num_per_page,
            "pageNumber": page_num
        }
        r = requests.get(endpoint, headers=headers, params=params)
        
        if "out of range" in r.text:
            if num_per_page == 1:
                break
            num_per_page = int(num_per_page / 2)
            # print("  halving num_per_page to", num_per_page)
            continue
        
        candidates = r.json()['data']['candidates']
        
        new_objids = [candidates[i]['id'] for i in range(len(candidates))]
        for new_objid in new_objids:
            if new_objid not in objids:
                objids += [new_objid]
                found_new = True
            else:
                # print("repeated source:", new_objid)
                pass
                
        if not found_new:
            # print("  no new sources")
            break
        
        page_num += 1        
        # print("  total", len(objids), objids[-3:-1], "\n")
        
        time.sleep(2)

    print("  Done querying for BTS rejects")
    rejects = pd.DataFrame(objids, columns=['ZTFID'])
    rejects.to_csv("data/base_data/rejects.csv", index=None)


def query_BTS_save_times():
    """
    Queries the time at which each BTS source was saved to the BTS fritz group

    Parameters
    ----------
    None

    Returns
    -------
    Nothing
    """
    print("  Querying for trues save times")
    trues = pd.read_csv("data/base_data/trues.csv", index_col=None)

    if "RCF_save_time" not in list(trues):
        trues["RCF_save_time"] = -1

    for i in tqdm(trues.index):
        objid = trues.loc[i, "ZTFID"]
        
        if trues.loc[i, "RCF_save_time"] > 0:
            continue

        endpoint = f"/api/sources/{objid}"
        r = requests.get(host+endpoint, headers=headers, params={})
        data = r.json()['data']

        if not r.ok:
            continue
        
        for group in data['groups']:
            if group['name'] == "Redshift Completeness Factor":
                trues.loc[i, "RCF_save_time"] = astrotime.Time(group['saved_at']).jd
        time.sleep(0.2)

    trues.to_csv("data/base_data/trues.csv", index=None)
    print("  Finished querying for BTS save times")        


# TODO add query_trigger_times()


def query_BTSSE(query_name, overwrite : bool = False):
    """
    Execute query to internal BTS sample explorer to fetch sources on predefined
    query. 

    Parameters
    ----------
    query_name : str
        name of query to be executed

    overwrite: bool (optional)
        whether or not to replace existing {query_name}.csv file

    Returns
    -------
    Nothing
    """
    if not (os.path.exists(f"data/base_data/{query_name}.csv") and not overwrite):
        with open(f"data/base_data/{query_name}.csv", "w") as f:
            f.write(requests.get(btsse_query_urls[query_name], 
                                 auth=(creds["btsse_username"], 
                                       creds["btsse_password"])).text)
            print("Queried and wrote", query_name)
    else:
        print(f"  {query_name} list already present")

    if query_name == "trues":
        query_BTS_save_times()


def compile_from_BTSSE(query_name, all_ZTFIDs, overwrite : bool = False):
    """
    Query ZTFIDs from BTS sample explorer and remove repeats from previous data

    Parameters
    ----------
    query_name : str
        name of query to be executed

    all_ZTFIDs : array
        running list of ZTFIDs already present in data
    
    overwrite : bool (optional)
        whether or not to replace existing {query_name}.csv file

    Returns
    -------
    query_df : pandas.DataFrame
        DataFrame containing information about query like query_df['ZTFID']

    all_ZTFIDs : array
        updated list of ZTFIDs present in data
    """
    query_BTSSE(query_name, overwrite)

    query_df = pd.read_csv(f"data/base_data/{query_name}.csv", index_col=None)

    query_df = query_df[~query_df["type"].isin(["duplicate", "duplicate?"])]
    query_df = query_df[~query_df['ZTFID'].isin(all_ZTFIDs)]

    all_ZTFIDs = np.concatenate((all_ZTFIDs, query_df["ZTFID"].to_numpy()))
    print(f"{query_name} done: {len(query_df)} new, {len(all_ZTFIDs)} total")

    return query_df, all_ZTFIDs


def compile_extIas(all_ZTFIDs):
    """
    Read list of Type-Ia SN ZTFIDs and remove repeats from previous data

    Parameters
    ----------
    all_ZTFIDs : array
        running list of ZTFIDs already present in data

    Returns
    -------
    extIas : pandas.DataFrame
        DataFrame containing ZTFIDs of new externally identified Type-Ia SNe

    all_ZTFIDs : array
        updated list of ZTFIDs present in data
    """
    
    extIas = pd.read_csv('data/base_data/external_Ias_full.csv')
    extIas.rename(columns={"ztfname": "ZTFID"}, inplace=True)

    # Some objects in this list are non-ZTF objects, remove them
    nonZTF = ~extIas['ZTFID'].str.contains('ZTF')
    extIas = extIas.drop(index=extIas['ZTFID'].index[nonZTF])

    extIas = extIas[~extIas['ZTFID'].isin(all_ZTFIDs)]

    all_ZTFIDs = np.concatenate((all_ZTFIDs, extIas["ZTFID"].to_numpy()))
    print(f"extIas done: {len(extIas)} new, {len(all_ZTFIDs)} total")

    return extIas, all_ZTFIDs


def compile_rejects(all_ZTFIDs, overwrite : bool = False):
    """
    Query ZTFIDs of unsaved BTS candidates from Fritz and remove repeats from 
    previous data

    Parameters
    ----------
    all_ZTFIDs : array
        running list of ZTFIDs already present in data
    
    overwrite : bool (optional)
        whether or not to replace existing rejects.csv file

    Returns
    -------
    rejects : pandas.DataFrame
        DataFrame containing ZTFIDs of new BTS rejects

    all_ZTFIDs : array
        updated list of ZTFIDs present in data
    """
    if not (os.path.exists(f"data/base_data/rejects.csv") and not overwrite):
        query_rejects()
    else:
        print("  rejects list already present")

    rejects = pd.read_csv(f"data/base_data/rejects.csv", index_col=None)

    rejects = rejects[~rejects['ZTFID'].isin(all_ZTFIDs)]

    all_ZTFIDs = np.concatenate((all_ZTFIDs, rejects["ZTFID"].to_numpy()))
    print(f"rejects done: {len(rejects)} new, {len(all_ZTFIDs)} total")

    return rejects, all_ZTFIDs


def compile_ZTFIDs(overwrite: bool = False):
    """
    Compiles extensive list of ZTFIDs with multiple queries that comprises the
    sources in the BTSbot training set. Each query writes a file to disk at
    BTSbot/data/base_data/{query_name}.csv

    Known queries are ["trues", "vars", "dims", "rejects", "extIas"]

    Parameters
    ----------
    overwrite: bool
        whether or not to requery and replace existing {query_name}.csv file

    Returns
    -------
    Nothing
    """

    all_ZTFIDs = np.array(())

    trues, all_ZTFIDs = compile_from_BTSSE("trues", all_ZTFIDs, overwrite)
    vars, all_ZTFIDs = compile_from_BTSSE("vars", all_ZTFIDs, overwrite)
    dims, all_ZTFIDs = compile_from_BTSSE("dims", all_ZTFIDs, overwrite)
    extIas, all_ZTFIDs = compile_extIas(all_ZTFIDs)
    rejects, all_ZTFIDs = compile_rejects(all_ZTFIDs, overwrite)

    # Objects to exclude, usually having mixed label or transient in reference
    objs_to_remove = [
        "ZTF18abdiasx", "ZTF21abyazip", "ZTF18aaadqua", "ZTF18aarrwmi", 
        "ZTF18aazijke", "ZTF18abncsdn", "ZTF18aaslhxt", "ZTF18aamigmk", 
        "ZTF18abdpvnd", "ZTF18aaqffyp"
    ]

    for query, query_name in zip([trues, vars, dims, extIas, rejects], 
                                 ["trues", "vars", "dims", "extIas", "rejects"]):
        query = query[~query["ZTFID"].isin(objs_to_remove)]
        query.to_csv(f"data/base_data/{query_name}.csv", index=None)

    print("Final number of objects:", len(all_ZTFIDs))
    print("Done compiling ZTFIDs")


if __name__ == "__main__":
    compile_ZTFIDs(overwrite=False)