from typing import Dict, Any
from datasets import Dataset, load_dataset


class TrainingData():
    def __init__(self, dataset_loc: str, val_set_size: int, tokenizer, cutoff_len) -> None:
        self.dataset_loc = dataset_loc
        self.val_set_size = val_set_size
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len

    def tokenize(self, prompt: str, **kwargs) -> Dict[str, Any]:
        result = self.tokenizer(
            prompt + self.tokenizer.eos_token,
            truncation=True,
            max_length=self.cutoff_len,
            padding=False,
            return_token_type_ids=False
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.cutoff_len
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        return result

    def prepare_data(self, use_eos_token=True, **kwargs) -> None:
        data = load_dataset("json", data_files=self.dataset_loc)

        train_val = data["train"].train_test_split(test_size=self.val_set_size, shuffle=True, seed=42)
        self.train_data = train_val["train"].shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token))
        self.val_data = train_val["test"].shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token))

    def generate_prompt(self, data_point, **kwargs):
        return self.make_prompt(
            "Find out about a patient",
            data_point["prompt"].strip(),
            data_point["completion"].strip()
        )
    
    def generate_and_tokenize_prompt(self, data_point, **kwargs):
        prompt = self.generate_prompt(data_point, **kwargs)
        return self.tokenize(prompt, **kwargs)
    
    @staticmethod
    def make_prompt(instruction, input_, output=""):
        return "{0}\n\n{1}\n{2}\n\n{3}\n{4}\n\n{5}\n{6}".format(
            "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.",
            "### Instruction:",
            instruction,
            "### Input:",
            input_,
            "### Response:",
            output
        )