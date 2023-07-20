import pandas as pd


def update_existing_dataset_with_text():
    """
    Run the few shot training example generator
    :return:
    """
    all_pico_entries = pd.read_csv("resources/redhot_data/st2_complete_release.csv")
    all_pico_entries_w_text_train = pd.read_csv(
        "resources/redhot_data/old/st2_train_inc_text_.csv"
    )
    all_pico_entries_w_text_test = pd.read_csv(
        "resources/redhot_data/old/st2_test_inc_text_.csv"
    )

    # concatenate the train and test dataframes
    all_pico_entries_w_text = pd.concat(
        [all_pico_entries_w_text_train, all_pico_entries_w_text_test]
    )

    # list all post_id values that are in both dataframes
    post_ids = list(
        set(all_pico_entries["post_id"].values).intersection(
            set(all_pico_entries_w_text["post_id"].values)
        )
    )
    print("Found {} post ids in both dataframes".format(len(post_ids)))
    # list all post_id values not in both dataframes
    post_ids_not_in_both = list(
        set(all_pico_entries["post_id"].values).symmetric_difference(
            set(all_pico_entries_w_text["post_id"].values)
        )
    )
    print("Found {} post ids not in both dataframes".format(len(post_ids_not_in_both)))

    # save all pico entries to file where the row contains post_ids_not_in_both
    all_pico_entries[
        all_pico_entries["post_id"].isin(post_ids_not_in_both)
    ].to_csv("resources/redhot_data/st2_complete_release_missing_text.csv", index=False)

if __name__ == "__main__":
    update_existing_dataset_with_text()
