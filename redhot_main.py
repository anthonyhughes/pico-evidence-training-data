import pandas as pd

from redhot import get_all_spans, build_prompt_completion_pairs_from_spans


def update_existing_dataset_with_text():
    """
    Run the few shot training example generator
    :return:
    """
    all_pico_entries = pd.read_csv(
        "resources/redhot_data/st2_complete_release_missing_text_inc_text.csv"
    )
    all_pico_entries_w_text_train = pd.read_csv(
        "resources/redhot_data/old/st2_train_inc_text_.csv"
    )

    # concatenate the train and test dataframes
    all_pico_entries_w_text = pd.concat(
        [all_pico_entries_w_text_train, all_pico_entries]
    )

    print(
        "Total {} post ids in for transformation".format(len(all_pico_entries_w_text))
    )

    final_claims = []

    # for each post id, find the corresponding text
    for post_id in all_pico_entries_w_text["post_id"].values:
        # get the text for the post id
        text = all_pico_entries_w_text.query(f'post_id == "{post_id}"')["text"].values[
            0
        ]

        claim = all_pico_entries_w_text.query(f'post_id == "{post_id}"')[
            "claim"
        ].values[0]

        annotations = all_pico_entries_w_text.query(f'post_id == "{post_id}"')[
            "stage2_labels"
        ].values[0]

        parsed_annotations = get_all_spans(annotations)

        prompts = build_prompt_completion_pairs_from_spans(
            post_id, category="", text=text, spans=parsed_annotations
        )

        prompt = " and ".join([prompts[key]["text"] for key in prompts.keys()])

        if prompt == "" or prompt.startswith("[deleted") or "by user]" in prompt:
            continue

        final_claims.append({"pico_elements": prompt.strip(), "claim": claim.strip()})

    final_claims = pd.DataFrame(final_claims)
    final_claims.to_csv("resources/redhot_data/claims_for_finetuning.csv", index=False)


if __name__ == "__main__":
    update_existing_dataset_with_text()
