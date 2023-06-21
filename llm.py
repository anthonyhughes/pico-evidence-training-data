import openai


def test_call_to_gpt3() -> str:
    """
    Test a call to GPT-3
    :return: gpt response
    """
    print('Testing a call to GPT-3')
    # create a chat completion
    chat_completion = openai.Completion.create(model="text-davinci-003",
                                               prompt="write me a social media post about the new iPhone 13.",
                                               temperature=0.9,
                                               max_tokens=56, )
    # print the chat completion
    return chat_completion.choices[0].text


def create_training_document(prompt: str, model: str = "text-davinci-003", max_tokens: int = 56) -> str:
    """
    Create a training document
    :param prompt: prompt
    :param model: openai LLM
    :param max_tokens: length of the response
    :return:
    """
    print('Creating a training document')
    # create a chat completion
    chat_completion = openai.Completion.create(model=model,
                                               prompt=prompt,
                                               temperature=0.9,
                                               max_tokens=max_tokens
                                               )
    print(chat_completion)
    # print the chat completion
    return chat_completion.choices[0].text


def create_chatgpt_training_document(prompt: str, max_tokens: int = 56) -> str:
    """
    Create a training document
    :param prompt: prompt
    :param max_tokens: length of the response
    :return:
    """
    print('Creating a training document')
    # create a chat completion
    messages = [
        {
            "role": "user",
            "content": f"{prompt}"
        }
    ]
    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                   messages=messages,
                                                   max_tokens=max_tokens
                                                   )
    print(chat_completion)
    # print the chat completion
    return chat_completion.choices[0].message.content


def fetch_available_models() -> str:
    """
    Fetch available models
    :return:
    """
    print('Fetching available models')
    response = openai.Model.list()
    # print(response)
    return ",".join([res['id'] for res in response['data']])
