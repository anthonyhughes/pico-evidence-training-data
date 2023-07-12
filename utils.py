from transformers import AutoModelForCausalLM

def print_parameters(model: AutoModelForCausalLM) -> None:
    trainable_parameters = 0
    all_parameters = 0
    for _, parameter in model.named_parameters():
        all_parameters += parameter.numel()
        if parameter.requires_grad:
            trainable_parameters += parameter.numel()
    
    print(f"Trainable parameters: {trainable_parameters}")
    print(f"All Parameters: {all_parameters}")
    print(f"Percentage trainable: {100 * trainable_parameters / all_parameters}")

def create_prompt(input: str = "I've read that the gaps diet can heal things like gerd") -> str:
    prompt = f"""
<patient>: I have am using diet to heal my gerd. What can I say to people?
<social media influencer>: {''}
"""
    print(prompt)
    return prompt

def create_prompt_for_training(question: str, answer: str) -> str:
    prompt = f"""
<doctor>: Tell me about your issues with {question}
<patient>: {answer}
"""
    print(prompt)
    return prompt