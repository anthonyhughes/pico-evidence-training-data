import json
from typing import List, Dict
from datasets import load_dataset
import pandas as pd
from transformers.pipelines.base import Dataset

SUBREDDIT_ID_TO_POPULATION = {
    "t5_2rtve": "lupus",
    "t5_2syer": "gout",
    "t5_2s3g1": "ibs",
    "t5_2tyg2": "Psychosis",
    "t5_395ja": "costochondritis",
    "t5_2saq9": "POTS",
    "t5_2s23e": "MultipleSclerosis",
    "t5_2s1h9": "Epilepsy",
    "t5_2qlaa": "GERD",
    "t5_2r876": "CysticFibrosis"
}

TARGET_CLASSES = [
    'claim',
    'per_exp',
    'claim_per_exp',
    'population',
    'intervention',
    'outcome'
]


def fetch_example_post() -> str:
    """
    Fetch an example post
    :return:
    """
    frame = pd.read_csv('resources/st1_train_inc_text_.csv')
    # select 2nd row of pandas frame
    frame = frame.sample(n=1)
    return frame['text'].head(2).values[0]


def clean_post_for_prompt(text: str) -> str:
    """
    Clean a post for a prompt
    :param text:
    :return:
    """
    # split into sentences
    sentences = text.split('.')
    sentences = list(filter(lambda sen: len(sen) > 0, sentences))[0:5]
    return ".".join(sentences)


def index_is_within_offsets(index: int, start_offset: int, end_offset: int) -> bool:
    """
    Check if an index is within a given span
    :param index:
    :param start_offset:
    :param end_offset:
    :return:
    """
    return start_offset <= index < end_offset


def get_span_for_index(index: int, spans: List) -> List:
    """
    Get the span for a given index
    :param index:
    :param spans:
    :return:
    """
    results = filter(lambda x: index_is_within_offsets(index, x['startOffset'], x['endOffset']), spans)
    return list(results)


def get_all_spans(annotation_spans: str) -> List[Dict]:
    """
    Get all annotation labels from a row from within the original corpus
    :param annotation_spans:
    :return:
    """
    data = json.loads(annotation_spans)
    return data[0]['crowd-entity-annotation']['entities']


def get_annotation_data(dataframe: pd.DataFrame,
                        target_row: int,
                        target_post: str,
                        target_col: str = 'stage1_labels',
                        ) -> Dict:
    if target_post:
        row = dataframe.query(f'post_id == "{target_post}"').iloc[0]
    else:
        row = dataframe.iloc[target_row]
    return {
        'annotation_spans': row[target_col],
        'text': row['text'],
        'subreddit_id': row['subreddit_id'],
        'post_id': row['post_id'],
    }


def build_prompt_completion_pairs_from_spans(post_id, category, text, spans) -> Dict:
    """
    Build a prompt completion pair from a given set of spans
    :param post_id:
    :param category:
    :param text:
    :param spans:
    :return:
    """
    results = {}
    for index, char in enumerate(text):
        filtered_spans = get_span_for_index(index=index, spans=spans)
        if len(filtered_spans) == 1 and filtered_spans[0]['label'] in TARGET_CLASSES:
            start = filtered_spans[0]['startOffset']
            end = filtered_spans[0]['endOffset']
            span_id = start + end
            if span_id in results:
                results[span_id]['text'] += char
            else:
                results[span_id] = {
                    'label': f'{filtered_spans[0]["label"]}',
                    'text': char
                }
    return results


def run_prompt_completion_pair_generator(
        all_entries_df: pd.DataFrame, target_row: int, target_col: str = 'stage1_labels',
        target_post: str = 'p61l9a') -> Dict:
    """
    Main function for running the annotation viewer
    :param target_post:
    :param target_col:
    :param target_row:
    :param all_entries_df: all text and annotations from a given corpus
    :return:
    """
    for_viewing = get_annotation_data(
        dataframe=all_entries_df, target_row=target_row, target_col=target_col, target_post=target_post
    )
    all_spans_for_viewing = get_all_spans(for_viewing['annotation_spans'])
    return build_prompt_completion_pairs_from_spans(post_id=for_viewing['post_id'],
                                                    category=for_viewing['subreddit_id'],
                                                    text=for_viewing['text'],
                                                    spans=all_spans_for_viewing)


def check_example_dataset():
    data = load_dataset("timdettmers/openassistant-guanaco")
    print(data)


def prep_hf_dataset() -> Dataset:
    dataset = load_dataset("json", data_files="resources/fine_tune_data.jsonl", split="train")
    print(dataset)
    print(dataset["prompt"][0], dataset["completion"][0])
    return dataset
