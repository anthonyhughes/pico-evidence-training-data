from llm import create_training_document, create_chatgpt_training_document


def build_davinci_training_example(prompt: str, model: str = "text-davinci-003"):
    """
    Build a training example for the davinci model
    :param model:
    :param prompt:
    :return:
    """
    result = create_training_document(prompt=prompt, model=model)
    to_log = f"""
        ########
        Prompt:
        {prompt}
        Result:
        {result}
        ########
        """

    with(open(f"./resources/{model}-output.txt", "a")) as f:
        f.writelines(to_log)


def build_chatgpt_training_example(prompt: str):
    """
    Build a training example for the chatgpt model
    :param prompt:
    :return:
    """
    model: str = "gpt-3.5-turbo"
    result = create_chatgpt_training_document(prompt=prompt)
    to_log = f"""
        ########
        Prompt:
        {prompt}
        Result:
        {result}
        ########
        """

    with(open(f"./resources/{model}-output.txt", "a")) as f:
        f.writelines(to_log)
