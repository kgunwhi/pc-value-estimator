import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

# Avoid bot detection
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
}

def get_soup(url):
    """
    Fetches and parses HTML from the given URL with polite delays and error handling.
    Returns a BeautifulSoup object if successful, None otherwise.
    """
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        time.sleep(random.uniform(1.5, 3.5))  # Random delay to mimic human behavior
        return BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None

def scrape_passmark_cpu(project_root):
    """
    Scrapes CPU Name, Score, Rank, Value Score, and Price from PassMark.
    Saves output as data/cpu_passmark.csv.
    """

    url = "https://www.cpubenchmark.net/cpu_list.php"
    soup = get_soup(url)
    if not soup: return

    table = soup.find('table', {'id': 'cputable'})
    rows = table.find_all('tr')

    cpu_data = []
    for row in rows[1:]:
        cols = row.find_all('td')
        if len(cols) < 5:
            continue

        name = cols[0].get_text(strip=True)
        score = cols[1].get_text(strip=True).replace(",", "")
        rank = cols[2].get_text(strip=True).replace(",", "")
        value = cols[3].get_text(strip=True)
        price = cols[4].get_text(strip=True).replace("$", "").replace("*", "").replace(",", "")

        try:
            cpu_data.append({
                "CPU": name,
                "PassMark_Score": int(score) if score != "NA" else None,
                "Rank": int(rank) if rank != "NA" else None,
                "ValueScore": float(value) if value != "NA" else None,
                "Price": float(price) if price != "NA" else None
            })
        except:
            continue

    df = pd.DataFrame(cpu_data)

    cpu_path = os.path.join(project_root, 'data', 'cpu_passmark.csv')
    df.to_csv(cpu_path, index=False)
    print(f"Scraped {len(df)} CPUs with scores, prices, and value.")

def scrape_passmark_gpu(project_root):
    """
    Scrapes GPU Name, Score, Rank, Value Score, and Price from PassMark.
    Saves output as data/gpu_passmark.csv.
    """
    os.makedirs("../data", exist_ok=True)

    url = "https://www.videocardbenchmark.net/gpu_list.php"
    soup = get_soup(url)
    if not soup: return

    table = soup.find('table', {'id': 'cputable'})
    rows = table.find_all('tr')

    gpu_data = []
    for row in rows[1:]:
        cols = row.find_all('td')
        if len(cols) < 5:
            continue

        name = cols[0].get_text(strip=True)
        score = cols[1].get_text(strip=True).replace(",", "")
        rank = cols[2].get_text(strip=True).replace(",", "")
        value = cols[3].get_text(strip=True)
        price = cols[4].get_text(strip=True).replace("$", "").replace("*", "").replace(",", "")

        try:
            gpu_data.append({
                "GPU": name,
                "PassMark_Score": int(score) if score != "NA" else None,
                "Rank": int(rank) if rank != "NA" else None,
                "ValueScore": float(value) if value != "NA" else None,
                "Price": float(price) if price != "NA" else None
            })
        except:
            continue

    df = pd.DataFrame(gpu_data)
    gpu_path = os.path.join(project_root, 'data', 'gpu_passmark.csv')
    df.to_csv(gpu_path, index=False)
    print(f"Scraped {len(df)} GPUs with scores, prices, and value.")

