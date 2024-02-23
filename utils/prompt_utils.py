import re


def extract_prompts(input_string):
    # Define the new regex pattern to include negative numbers
    pattern = r"\"(-?\d+)\.([^\"]+)\":\s*\"([^\"]+)\""

    # Find all matches using the pattern
    matches = re.findall(pattern, input_string)

    # Convert matches to a list of tuples (number, name, text)
    result = [(int(number), name.strip(), text)
              for number, name, text in matches]

    return result
