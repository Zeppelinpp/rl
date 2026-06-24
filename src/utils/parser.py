import re


def get_thinking_content(text: str):
    """
    Extrat the <think> </think> content from the text, including the <think> </think> tags
    :param text: the text to extract the thinking content from
    :return: the extracted thinking content
    """
    thinking_content = re.split(r"(<think>.*?</think>)", text, flags=re.DOTALL)
    return [part for part in thinking_content if part]


if __name__ == "__main__":
    text = "<think>Ok, I am a human </think> I'm a human"
    print(get_thinking_content(text))
