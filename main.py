import argparse
import json

import pandas as pd
import openai

from llm import fetch_available_models
from prompt import build_prompt, build_prompt_for_fine_tuned_model
from redhot import fetch_example_post, clean_post_for_prompt, run_prompt_completion_pair_generator
from train import run_few_shot_trainer
from training_data import build_davinci_training_example, build_chatgpt_training_example
from trialstreamer import fetch_example_trialstreamer_elements, transform_elements_for_prompt

openai.api_key = "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"


def run_zero_shot_training_generator():
    """
    Run the zero shot training example generator
    :return:
    """
    elements = fetch_example_trialstreamer_elements()
    updated_elements = transform_elements_for_prompt(elements)

    sm_post = fetch_example_post()
    cleanm_post = clean_post_for_prompt(sm_post)
    PROMPT_EXTRAS = 'Include alternative names for the medical concepts. ' \
                    'Include spelling and grammatical mistakes, and use colloquial language.'

    print("Building prompt")
    prompt = build_prompt(
        social_media_post=cleanm_post,
        medical_abstract=updated_elements[0],
        populations=updated_elements[1],
        interventions=updated_elements[2],
        outcomes=updated_elements[3],
        additional_instructions=PROMPT_EXTRAS
    )

    print("Generating davinci example")
    build_davinci_training_example(prompt)

    print("Generating chatgpt example")
    build_chatgpt_training_example(prompt)


def run_automated_few_shot_training_generator():
    """
    Run the few shot training example generator
    :return:
    """
    all_claim_entries = pd.read_csv('./resources/redhot_corpora/st1_train_inc_text_.csv')
    all_pico_entries = pd.read_csv('./resources/redhot_corpora/st2_train_inc_text_.csv')
    # list all post_id values that are in both dataframes
    post_ids = list(
        set(all_claim_entries['post_id'].values).intersection(
            set(all_pico_entries['post_id'].values))
    )
    print("Found {} post ids in both dataframes".format(len(post_ids)))
    for post_id in post_ids:
        completions = run_prompt_completion_pair_generator(
            all_entries_df=all_claim_entries,
            target_row=0,
            target_col='stage1_labels',
            target_post=post_id
        )

        prompts = run_prompt_completion_pair_generator(
            all_entries_df=all_pico_entries,
            target_row=0,
            target_col='stage2_labels',
            target_post=post_id
        )
        for completion in completions.keys():
            picos = []
            for prompt_key in prompts.keys():
                picos.append(prompts[prompt_key]['text'])
            pair = {
                'prompt': ",".join(picos),
                'completion': ' ' + completions[completion]['text']
            }
            if picos:
                with(open('./resources/fine_tune_data.jsonl', 'a')) as f:
                    f.write(json.dumps(pair) + '\n')


def run_fine_tune_generator():
    """
    Run the zero shot training example generator
    :return:
    """
    elements = fetch_example_trialstreamer_elements()
    updated_elements = transform_elements_for_prompt(elements)

    print("Building prompt")
    prompt = build_prompt_for_fine_tuned_model(
        populations=updated_elements[1],
        interventions=updated_elements[2],
        outcomes=updated_elements[3]
    )

    print("Generating medical claim")
    # previous model = ada:ft-personal-2023-06-19-16-17-15
    build_davinci_training_example(prompt, model='davinci:ft-personal-2023-06-20-11-31-16')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Training data generator',
        description='Generate training data for evidence retrieval',
        epilog='Example: python main.py --script zero-shot')
    parser.add_argument('--script', help='The script to run', required=True)
    args = parser.parse_args()
    if args.script == 'zero-shot-generator':
        run_zero_shot_training_generator()
    elif args.script == 'few-shot-example-generator':
        run_automated_few_shot_training_generator()
    elif args.script == 'few-shot-train':
        run_few_shot_trainer()
    elif args.script == 'fine-tune-generator':
        run_fine_tune_generator()
    elif args.script == 'list-models':
        models = fetch_available_models()
        print(models)
    else:
        print('Unknown script')
