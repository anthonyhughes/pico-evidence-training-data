import argparse
import glob
import json
import re
from typing import Tuple
import pandas as pd
from tqdm import tqdm

EXCLUSION_LIST = [
    "Patient",
    "Patients",
    "Humans",
    "With",
    "Score",
    "Scores",
    "Mean",
    "Total",
    "Containing",
    "Early",
    "Method"
]

LOWER_CASE_LIST = [e.lower() for e in EXCLUSION_LIST]

def fetch_example_trialstreamer_elements(
    csv_location: str = "resources/trialstreamer/trialstreamer-update-pubmed-2022-06-20.csv",
) -> Tuple:
    """
    Fetch the elements from the trialstreamer csv file
    """
    print("Fetching example trialstreamer elements")
    frame = pd.read_csv(csv_location)
    frame = frame.sample(n=1)
    return (
        frame["ab"].head(1).values[0],
        frame["population_mesh"].head(1).values[0],
        frame["interventions_mesh"].head(1).values[0],
        frame["outcomes_mesh"].head(1).values[0],
        frame["pmid"].head(1).values[0],
    )


def fetch_trialstreamer_elements(
    csv_location: str = "resources/trialstreamer/trialstreamer-update-pubmed-2022-06-20.csv",
) -> list[Tuple]:
    """
    Fetch the elements from the trialstreamer csv file
    """
    print("Fetching example trialstreamer elements")
    frame = pd.read_csv(csv_location)
    frame_elements = []
    print(f"Transforming {len(frame)} rows for prompting")
    for index, row in tqdm(frame.iterrows()):
        frame_elements.append(
            (
                row["population_mesh"],
                row["interventions_mesh"],
                row["outcomes_mesh"],
                row["pmid"],
                row["ab"]
            )
        )
    return frame_elements 


def contains_digit(input: str) -> bool:
    return any(char.isdigit() for char in input)


def replace_double_quotes_within_word(input_str):
    """
    Replace double quotes within a word with single quotes
    """
    input_str = input_str.replace("'", '"')
    pattern = r'\b(\w+)"(\w+|-|,| )\b'

    # Define a function that replaces the double quotes with single quotes
    def replace_single(match):
        return match.group(1) + "'" + match.group(2)

    # Use re.sub() to perform the replacement
    result_str = re.sub(pattern, replace_single, input_str)
    return result_str


def transform_list(
    tlist: str, limit: int = 5, abstract_id: str = "", separator: str = " and "
) -> str:
    """
    Transform a list of elements into a comma separated string
    """
    try:
        try:
            es = json.loads(tlist.replace("'", '"'))
        except json.decoder.JSONDecodeError as e:
            try:
                # Replace single quotes around keys with double quotes                
                tlist = replace_double_quotes_within_word(tlist)
                es = json.loads(tlist)
            except json.decoder.JSONDecodeError as e:
                # print(f"JSONDecodeError for abstract {abstract_id}")
                # print(tlist)
                # print(replace_double_quotes_within_word(tlist))
                return ""               
        # filter any elements without a cui_str
        es = filter(lambda x: "cui_str" in x, es)
        # extract all cui_str values only
        es = [e["cui_str"] for e in es]
        # filter certain generic concepts
        es = filter(lambda x: x not in EXCLUSION_LIST, es)
        es = filter(lambda x: x not in LOWER_CASE_LIST, es)
        # strip any trailing whitespace
        es = [e.strip().lower() for e in es]
        # remove any content in brackets
        es = [e.split("(")[0] for e in es]
        # filter any digits (used to represent patient participant count)
        es = filter(lambda x: x.isdigit() is False, es)
        es = filter(lambda x: contains_digit(x) is False, es)
        # filter any long concepts
        es = filter(lambda x: len(x.split(" ")) <= 3, es)
        # filter qualifier values (used to represent the mesurements of the outcome)
        es = filter(lambda x: "(qualifier value)" not in x, es)

        es = list(es)[:limit]
        return f"{separator}".join(es)
    except KeyError as e:
        print(f"KeyError for abstract {abstract_id}")
        return ""


def transform_elements_for_prompt(elements: Tuple) -> Tuple:
    # print("Transforming elements for prompt")
    abstract_id = elements[3]
    return (
        transform_list(elements[0], 2, abstract_id),
        transform_list(elements[1], 2, abstract_id),
        transform_list(elements[2], 2, abstract_id),
        abstract_id,
    )


def updates_to_prompts(frame: pd.DataFrame) -> pd.DataFrame:
    # drop rows with population or intervention missing
    frame = frame[frame["population"] != ""]
    frame = frame[frame["intervention"] != ""]
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, default="trialstreamer")
    args = parser.parse_args()

    if args.script == "full-trialstreamer-extract":
        folder_location = "resources/trialstreamer/"
        target_files = [f for f in glob.glob(f"{folder_location}/publications/*.csv")]
        # target_files = [f"{folder_location}/trialstreamer-update-pubmed-2022-06-06.csv"]
        all_elements = []
        all_abstracts = []
        for trialstreamer_doc in target_files:
            print(f"Processing {trialstreamer_doc}")
            document_data = fetch_trialstreamer_elements(trialstreamer_doc)
            elements = [transform_elements_for_prompt(el) for el in document_data]
            print(f"Processed {trialstreamer_doc}")
            all_elements.extend(elements)
            id_to_abstract = [(el[3], el[4]) for el in document_data]
            all_abstracts.extend(id_to_abstract)
            print(f"Total elements: {len(elements)}")        
            print(f"Current total elements: {len(all_elements)}")        
        final_df = pd.DataFrame(
            all_elements,
            columns=["population", "intervention", "outcome", "pmid"],
        )
        print(f"Total elements: {len(final_df)}")
        final_df = updates_to_prompts(final_df)
        print(f"Total elements: {len(final_df)}")

        abstract_df = pd.DataFrame(
            all_abstracts,
            columns=["pmid", "abstract"],
        )
        final_df.to_csv(
            f"{folder_location}/final/prompts-for-generation.csv", index=False
        )
        abstract_df.to_csv(
            f"{folder_location}/final/abstracts-for-retrieval.csv", index=False
        )
    elif args.script == "example-trialstreamer-extract":
        elements = fetch_example_trialstreamer_elements()
        elements = transform_elements_for_prompt(elements)
        print(elements)
