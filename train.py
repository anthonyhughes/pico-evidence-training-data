import time
import os
import openai

training_file = "resources/training_data.jsonl"
validation_file = "resources/validation_data.jsonl"

tuning_args = {
    "training_file": training_file,
    "validation_file": validation_file,
    "model": "davinci",
    "n_epochs": 15,
    "batch_size": 3,
    "learning_rate_multiplier": 0.3
}


def run_few_shot_trainer() -> None:
    # Create the fine-tuning job
    openai.api_key = "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"
    fine_tuning_job = openai.FineTune.create(**tuning_args)
    print('Wait for the fine-tuning job to complete')
    job_id = fine_tuning_job["id"]
    print(f"Fine-tuning job created with ID: {job_id}")

    while True:
        fine_tuning_status = openai.FineTune.get_status(job_id)
        status = fine_tuning_status["status"]
        print(f"Fine-tuning job status: {status}")

        if status in ["completed", "failed"]:
            break

        time.sleep(60)
