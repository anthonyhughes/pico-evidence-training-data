import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

from redhot import check_example_dataset, prep_hf_dataset

model_id = "tiiuae/falcon-7b"


def build_pipeline(tokenizer):
    return transformers.pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )


def generate_sequence(fal_pipeline, tokenizer: AutoTokenizer, prompt: str):
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
    return AutoTokenizer.from_pretrained(model_id, cache_dir="/data/ahughes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Training data generator',
        description='Generate training data for evidence retrieval using Falcon',
        epilog='Example: python main.py --script zero-shot')
    parser.add_argument('--script', help='The script to run', required=True)
    args = parser.parse_args()
    if args.script == 'dataset-test':
        print('Running dataset generator')
        hf_dataset = prep_hf_dataset()
    elif args.script == 'generate-example-sequence':
        print('Running test')
        tokenizer = build_tokenizer()
        pipeline = build_pipeline(tokenizer=tokenizer)
        generate_sequence(fal_pipeline=pipeline, tokenizer=tokenizer,
                          prompt="You are a social media bot. "
                                 "Can you produce a very personal sounding medical claim written pretending to be a "
                                 "person with a medical condition?")
    else:
        print('Unknown script')




