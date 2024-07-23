import re

def extract_number(phrase):
    match = re.search(r'\d+', phrase)
    return match.group(0) if match else -1