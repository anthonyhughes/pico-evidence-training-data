def build_prompt(social_media_post: str, medical_abstract: str, populations: str, interventions: str,
                 outcomes: str, additional_instructions: str) -> str:
    """
    Build a prompt for an LLM
    :param social_media_post:
    :param medical_abstract:
    :param populations:
    :param interventions:
    :param outcomes:
    :param additional_instructions:
    :return:
    """
    print('Building prompt')
    return f'''    
    {'This is an example of personal medical post from social media:' + social_media_post if social_media_post else ''}    
    {"Medical abstract:" + medical_abstract if medical_abstract else ''}    
    {populations}
    Interventions:
    {interventions}
    Outcomes:
    {outcomes} 
    Using this information, write a short personal medical claim written by a reddit or twitter user.
    {additional_instructions if additional_instructions else ''}     
    '''


def build_prompt_for_fine_tuned_model(populations: str, interventions: str, outcomes: str) -> str:
    """
    Build a prompt for a custom LLM
    :param populations:
    :param interventions:
    :param outcomes:
    :return:
    """
    print('Building prompt')
    # return f'''{populations}, {interventions}, {outcomes}'''
    return f'''cardiac surgical patients,ventilation\n\n###\n\n'''


def test_a_prompt():
    print('Testing a prompt')
    instructions = [
        'Include alternative names for the medical concepts',
        'Include spelling and grammatical mistakes, and use colloquial language'
    ]
    out = build_prompt('test', 'abstract', 'populations', 'interventions', 'outcomes', '.'.join(instructions))
    print(out)
