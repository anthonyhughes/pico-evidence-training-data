import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from data import TrainingData
from utils import print_parameters, create_prompt_for_training, add_new_column
from train import build_training_arguments, build_trainer
from redhot import check_example_dataset, prep_hf_dataset

model_id = "tiiuae/falcon-7b"


def build_pipeline(tokenizer):
    print("Building pipeline")
    return transformers.pipeline(
        "text-generation",
        model="/data/ahughes",
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )


def generate_sequence(fal_pipeline, tokenizer: AutoTokenizer, prompt: str) -> None:
    sequences = fal_pipeline(
        prompt,
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")


def build_tokenizer():
    print("Building tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir="/data/ahughes", return_token_type_ids=False
    )
    tokenizer.truncation_side = "left"
    tokenizer.bos_token_id = None
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prep_model_for_training() -> AutoModelForCausalLM:
    print("Prepping model")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        # device_map="auto",
        cache_dir="/data/ahughes",
        # load_in_8bit=True,
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()
    return model


def run_inference(model, prompt, tokenizer):
    generation_config = model.generation_config
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = 52
    generation_config.do_sample = True
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.top_k = 10
    generation_config.top_p = 0.1
    generation_config.temperature = 0.1

    print("Config for generation")
    # print(generation_config)

    encoding = tokenizer(
        prompt + tokenizer.eos_token,
        truncation=True,
        padding=False,
        return_token_type_ids=False,
        return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        outputs = model.generate(
            encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config,
        )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def finalise_data(hf_dataset, tokenizer):
    def update_sample_as_prompt(data_sample):
        full_prompt = create_prompt_for_training(
            data_sample["prompt"], data_sample["completion"]
        )
        tokenised_full_prompt = tokenizer(full_prompt, padding=True, truncation=True)
        data_sample["input_ids"] = tokenised_full_prompt.input_ids
        data_sample["attention_mask"] = tokenised_full_prompt.attention_mask
        return data_sample

    hf_dataset = hf_dataset.shuffle().map(update_sample_as_prompt)
    print(hf_dataset)
    return hf_dataset


if __name__ == "__main__":
    print("Starting!")
    parser = argparse.ArgumentParser(
        prog="Training data generator",
        description="Generate training data for evidence retrieval using Falcon",
        epilog="Example: python main.py --script zero-shot",
    )
    parser.add_argument("--script", help="The script to run", required=True)
    args = parser.parse_args()
    if args.script == "train":
        print("Running train")
        tokenizer = build_tokenizer()
        hf_dataset = TrainingData(
            dataset_loc="./resources/fine_tune_data.jsonl",
            val_set_size=100,
            tokenizer=tokenizer,
            cutoff_len=256,
        )

        hf_dataset.prepare_data()
        model = prep_model_for_training()

        training_arguments = build_training_arguments(eval_steps=50)
        trainer = build_trainer(
            model=model,
            dataset=hf_dataset,
            training_args=training_arguments,
            tokenizer=tokenizer,
        )

        trainer.train()
        trainer.save_model("/data/ahughes/falcon-finetune-v2")
    elif args.script == "generate-example-sequence":
        print("Running test")
        tokenizer = build_tokenizer()
        pipeline = build_pipeline(tokenizer=tokenizer)
        generate_sequence(
            fal_pipeline=pipeline,
            tokenizer=tokenizer,
            prompt="You are a social media bot. "
            "Can you produce a very personal sounding medical claim written pretending to be a "
            "person with a medical condition?",
        )
    elif args.script == "inference-finetuned":
        print("Running test")
        model = AutoModelForCausalLM.from_pretrained(
            "/data/ahughes/falcon-finetuned",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        tokenizer = build_tokenizer()
        prompt = create_prompt_for_training(prompt="diabetes", completion="")
        run_inference(tokenizer=tokenizer, model=model, prompt=prompt)
    else:
        print("Unknown script")

    print("Complete!")
