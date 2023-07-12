import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from utils import print_parameters

from redhot import check_example_dataset, prep_hf_dataset

model_id = "tiiuae/falcon-7b"

def build_pipeline(tokenizer):
    print('Building pipeline')
    return transformers.pipeline(
        "text-generation",
        model=model_id,
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


def build_tokenizer() -> AutoTokenizer:
    print('Building tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/data/ahughes")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def prep_model_for_training() -> AutoModelForCausalLM:
    print('Prepping model')
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    return model


def run_inference(model, prompt, tokenizer):
    generation_config = model.generation_config
    generation_config.max_new_tokens = 50
    generation_config.temperature = 0.7
    generation_config.top_p = 0.7
    generation_config.num_return_sequences = 1
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    print('Config for generation')
    print(generation_config)

    encoding = tokenizer(prompt, return_tensors='pt').to('cuda')
    with torch.inference_mode():
        outputs = model.generate(
            encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config
        )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

def finalise_data(data_sample):
    full_prompt =  

if __name__ == "__main__":
    print('Starting!')
    parser = argparse.ArgumentParser(
        prog='Training data generator',
        description='Generate training data for evidence retrieval using Falcon',
        epilog='Example: python main.py --script zero-shot')
    parser.add_argument('--script', help='The script to run', required=True)
    args = parser.parse_args()
    if args.script == 'train':
        print('Running train')
        hf_dataset = prep_hf_dataset()     
        data = finalise_data()()
        tokenizer = build_tokenizer() 
        model = prep_model_for_training()  
        print_parameters(model)
        prompt = create_prompt()
        run_inference(model=model, tokenizer=tokenizer, prompt=prompt)
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

    print('Complete!')
