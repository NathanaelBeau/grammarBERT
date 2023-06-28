def remove_comment(text):
    text = '\n'.join(filter(lambda x: x, text.split('\n')))  # Remove spaces between lines
    return text
