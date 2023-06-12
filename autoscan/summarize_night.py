#!/usr/bin/env python3
import sys, glob, pandas as pd
from weasyprint import HTML
import astropy.time as astrotime

if sys.platform == "darwin":
    base_path = "/Users/nabeelr/Desktop/School/ZTF Research/BNB-classifier/autoscan/"
else:
    base_path = "/projects/b1094/rehemtulla/BNB-classifier/autoscan/"

def summarize_night():
    """
    Collect list of all sources saved today (UTC) and save a summary to a PDF
    Use with output from autoscan_30min.py

    Intended to be run every day at 11AM CT with cron:
        * 11 * * * path/python3 path/summarize_night.py >> path/log.log 2>&1

    Parameters
    ----------
    None

    Returns
    -------
    Nothing
    """

    now = astrotime.Time.now()
    date = now.strftime('%h%d')  # formatted as "MonDD"

    # Collect all of tonight's saved sources written by autoscan
    files = glob.glob(f"nightly_summaries/{date}/*")

    nights_cand = pd.DataFrame()

    for file in files:
        nights_cand = pd.concat((nights_cand, pd.read_csv(file)))
        
    nights_cand = nights_cand.reset_index(drop=True)
    
    # Make dates relative to present
    nights_cand.iloc[:, 1:] = nights_cand.iloc[:, 1:].subtract(now.jd)
    
    # Clean up for aesthetics
    nights_cand = nights_cand.round(3)
    nights_cand = nights_cand.fillna("")

    # Hyperlink objectIds to their fritz pages
    for idx in nights_cand.index:
        objid = nights_cand.loc[idx, 'objectId']
        nights_cand.loc[idx, 'objectId'] =  f"<a href='https://fritz.science/source/{objid}'>{objid}</a>"

    # Represent summary as HTML
    head_text = f"BTSbot {date}"
    intro_text = f"BTSbot summary for {date} UTC"
    html_table = nights_cand.to_html(index=False, escape=False)
    outro_text = f"Days ago relative to {now.strftime('%h %d %H:%M UTC')}"

    html_content = f"""
    <html>
        <head>
            {head_text}
        </head>
        <body>
            <p>{intro_text}</p>
            {html_table}
            <p>{outro_text}</p>
        </body>
    </html>
    """

    HTML(html_content).write_pdf(f"{base_path}nightly_summaries/{date}/{date}_summary.pdf")
    print(html_content)

if __name__ == "__main__":
    summarize_night()