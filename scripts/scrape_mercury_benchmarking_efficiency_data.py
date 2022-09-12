import argparse
from bs4 import BeautifulSoup
import requests
import json
import os
import pandas as pd
from selenium import webdriver
import time
import tqdm

BASE_URL = "https://crfm-models.stanford.edu/static/benchmarking.html?runs"
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
        url = f"https://crfm-models.stanford.edu/static/benchmark_output/runs/latest/{run}/stats.json"
        r = requests.get(url)
        try:
            data = {}
            stats = json.loads(r.text)
            for metric in stats:
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
            print(f"Could not parse stats for run {run} at {url}")
            continue

    return all_data


def save_per_instance_stats(runs):
    for run in tqdm.tqdm(runs):
        directory = f"benchmark_output/runs/latest/{run}"
        if os.path.isdir(directory):
            continue
        url = f"https://crfm-models.stanford.edu/static/benchmark_output/runs/latest/{run}/per_instance_stats.json"
        r = requests.get(url)
        try:
            stats = json.loads(r.text)
            os.makedirs(directory, exist_ok=True)
            with open(f"{directory}/per_instance_stats.json", 'w') as f:
                f.write(r.text)
        except Exception as e:
            print(f"Could not parse stats for run {run} at {url}")
            continue


def save(data, keys, output_file):
    df = pd.DataFrame(data, columns=["run", *keys])
    df.to_csv(output_file, sep=",", index=False)


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
    if args.save_aggregated_stats:
        data = parse(runs, keys)
        save(data, keys, args.output_file)
    if save_per_instance_stats:
        save_per_instance_stats(runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape CRFM benchmarking results")
    parser.add_argument(
        "--output_file", type=str, default="mercury_benchmarking_inference_data.csv"
    )
    parser.add_argument("--save-aggregated-stats", action='store_true')
    parser.add_argument("--save-per-instance-stats", action='store_true')
    args = parser.parse_args()
    main(args)
