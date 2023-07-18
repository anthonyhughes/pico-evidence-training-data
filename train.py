from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

def build_training_arguments(eval_steps: int):
    print('Set trainging arguments')
    ta = TrainingArguments(
        per_device_train_batch_size=1,
        learning_rate=3e-4,
        fp16=True,
        fp16_full_eval=True,
        logging_steps=5,
        gradient_accumulation_steps=4,
        save_total_limit=3,
        output_dir='/data/ahughes/falcon-finetune-v2',
        num_train_epochs=3,
        warmup_steps=5,
        optim="adamw_torch",
        eval_steps=eval_steps
    )
    print('Training arguments', ta)
    return ta

def build_trainer(model, dataset, training_args, tokenizer):
    print('Building trainer')
    train = Trainer(
        model=model,
        train_dataset=dataset.train_data,
        eval_dataset=dataset.val_data,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    print('Trainer', train)
    return train