import argparse
from bs4 import BeautifulSoup
import requests
import json
import pandas as pd
from selenium import webdriver
import time
import tqdm

BASE_URL = "https://crfm-models.stanford.edu/static/benchmarking.html"
DELAY = 60


def scrape():
    driver = webdriver.Safari()

    driver.get(BASE_URL)

    print(f"Waiting {DELAY} seconds for results to render...")
    time.sleep(DELAY)
    innerHTML = driver.execute_script("return document.body.innerHTML")
    root = BeautifulSoup(innerHTML, "lxml")

    links = root.html.body.find("div", id="main").find_all("a")
    runs = [link.text for link in links]

    return runs


def parse(runs, keys):
    all_data = []

    for run in tqdm.tqdm(runs):
        url = f"https://crfm-models.stanford.edu/static/benchmark_output/runs/latest/{run}/metrics.json"
        r = requests.get(url)
        try:
            data = {}
            metrics = json.loads(r.text)
            for metric in metrics:
                name = metric["name"]["name"]
                if name in keys and (name != "exact_match" or metric["name"]["k"] == 1):
                    if len(metric["values"]) == 0:
                        data[name] = None
                    else:
                        data[name] = metric["values"][0]
            for key in keys:
                if key not in data:
                    data[key] = None
            all_data.append((run, *[data[key] for key in keys]))
        except Exception as e:
            print(f"Could not parse metrics for run {run} at {url}")
            print(r.text)
            continue

    return all_data


def save(data, keys, output_file):
    df = pd.DataFrame(data, columns=["run", *keys])
    df.to_csv(output_file, sep="\t", index=False)


def main(args):
    keys = [
        "inference_runtime",
        "inference_idealized_runtime",
        "inference_runtime_discrepancy",
        "num_prompt_tokens",
        "num_output_tokens",
        "finish_reason_length",
        "finish_reason_stop",
        "finish_reason_endoftext",
        "finish_reason_unknown",
        "exact_match",
    ]

    runs = scrape()
    data = parse(runs, keys)
    save(data, keys, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape CRFM benchmarking results")
    parser.add_argument(
        "--output_file", type=str, default="mercury_benchmarking_inference_data.tsv"
    )
    args = parser.parse_args()
    main(args)
