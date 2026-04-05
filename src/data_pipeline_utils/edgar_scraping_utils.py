import pandas as pd
import re
import yfinance as yf
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from edgar import *
from lxml import html
import asyncio
import aiohttp

set_identity("kamendd@hotmail.com")

def extract_mdna_text(accession):
    """
    Utilize edgartools to extract text for MD&A from 10-Q filing
    """
    filing = find(accession)
    
    if filing is None:
        return pd.NA
        
    try:
        mda_text = filing.obj()["Item 2"]
        return mda_text if pd.notna(mda_text) and mda_text != "" else pd.NA
    except (KeyError, AttributeError, ValueError):
        return pd.NA

def extract_press_release(accession):
    filing = find(accession)
    
    if filing is None:
        return pd.NA
        
    try:
        pr_text = filing.obj().get("Item 2.02")
        return pr_text if pd.notna(pr_text) and pr_text != "" else pd.NA
    except (KeyError, AttributeError, ValueError):
        return pd.NA

def extract_press_release(accession):
    """
    Utilize edgartools to extract text for Exhibit 99.1 Press Release from 8-K filing
    """
    set_identity("kamendd@hotmail.com")
    filing = find(accession)
    eight_k = filing.obj()
    if eight_k.has_press_release:
        releases = eight_k.press_releases
        pr = releases[0]
        return pr.text()
    else:
        return pd.NA

def extract_accession_number(url):
    """
    Take an SEC filing link, decompose via stripping by "/"
    Extracting from the list the url portion known as accession number
    """
    tokens = url.split('/')
    accession_number = tokens[-1].replace('-index.htm', '')
    return accession_number

def process_filing_row(row):
    """
    Method to apply the correct text extraction method depending on filing type
    """
    accession = row['accession']
    filing_type = row['Filing Type']
    text = None
    
    if filing_type == '10-Q':
        text = extract_mdna_text(accession)
    elif filing_type == '8-K':
        text = extract_press_release(accession)

    if pd.notna(text) and text != "":
        return text
    else:
        return pd.NA

async def check_compliance_async(session, row, sem, rate_lock, rate_state, index):
    """
    Asynchronous method scrape the filing web site for presence of Item 2.02 in an 8-K report.
    This confirms that the checked 8-K report is indeed an earnings announcement.
    The method returns a boolean.
    The method allows using a semaphore system and multiple workers to work close to the EDGAR
    datavase official 10 calls/second rate.
    """
    f_type = str(row['Filing Type']).upper()
    url = row['SEC Link']
    headers = {'User-Agent': 'SoftUni Student Project Kamen Dimitrov (kamendd@hotmail.com)'}

    # 10-Qs are instant and don't need the network
    if '10-Q' in f_type:
        return True
    
    # 8-Ks: Wait for your specific "timeslot" to avoid the 429 error
    async with sem:
        # --- Global rate limiting block ---
        async with rate_lock:
            loop = asyncio.get_running_loop()
            now = loop.time()

            min_interval = 0.12
            elapsed = now - rate_state['last_call']

            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)

            rate_state['last_call'] = loop.time()
        # --- End rate limiting block ---

        try:
            async with session.get(url, headers=headers, timeout=15) as response:
                if response.status == 200:
                    text = await response.text()
                    return bool(re.search(r'Item\s*(?:&nbsp;)?\s*2\.02', text, re.IGNORECASE))
                else:
                    return f"Error {response.status}"
        except Exception as e:
            return f"Fail: {str(e)}"


async def run_compliance_batch(df):
    """
    This method synchronizes the execution of all requests in a batch asynchronously.
    The method ensures that the check_compliance method is used without breaching the 
    10 requests/second rate limit.
    """
    
    sem = asyncio.Semaphore(9)
    connector = aiohttp.TCPConnector(limit=9)

    rate_lock = asyncio.Lock()
    rate_state = {'last_call': 0.0}

    data_list = df.to_dict('records')

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i, row in enumerate(data_list):
            tasks.append(
                check_compliance_async(session, row, sem, rate_lock, rate_state, i)
            )

        print("Starting SEC Compliance Check... this will take a few minutes.")
        results = await asyncio.gather(*tasks)
        return results