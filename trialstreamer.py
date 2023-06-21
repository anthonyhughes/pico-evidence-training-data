import json
from typing import Tuple

import pandas as pd


def fetch_example_trialstreamer_elements() -> Tuple:
    print('Fetching example trialstreamer elements')
    frame = pd.read_csv('resources/trialstreamer-update-pubmed-2022-06-20.csv')
    frame = frame.sample(n=1)
    return (
        frame['ab'].head(1).values[0],
        frame['population_mesh'].head(1).values[0],
        frame['interventions_mesh'].head(1).values[0],
        frame['outcomes_mesh'].head(1).values[0]
    )


def transform_list(list: str, limit: int = 3) -> str:
    es = json.loads(list.replace("'", '"'))
    es = [e['cui_str'] for e in es][:limit]
    return ",".join(es)


def transform_elements_for_prompt(elements: Tuple) -> Tuple:
    print('Transforming elements for prompt')
    return (
        elements[0],
        transform_list(elements[1]),
        transform_list(elements[2]),
        transform_list(elements[3]),
    )
