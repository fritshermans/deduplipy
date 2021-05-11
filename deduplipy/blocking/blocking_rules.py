import re


def whole_field(x):
    x_trimmed = x.strip()
    if len(x_trimmed):
        return x_trimmed
    else:
        return None


def first_word(x):
    x_trimmed = x.strip()
    if len(x_trimmed):
        return x_trimmed.split()[0]
    else:
        return None


def first_two_words(x):
    x_trimmed = x.strip()
    if len(x_trimmed):
        return " ".join(x_trimmed.split()[:2])
    else:
        return None


def first_three_letters(x):
    x_trimmed = x.strip()
    if len(x_trimmed):
        return x_trimmed[:3]
    else:
        return None


def first_four_letters(x):
    x_trimmed = x.strip()
    if len(x_trimmed):
        return x_trimmed[:4]
    else:
        return None


def first_five_letters(x):
    x_trimmed = x.strip()
    if len(x_trimmed):
        return x_trimmed[:5]
    else:
        return None


def first_three_letters_no_space(x):
    x = x.replace(' ', '')
    if len(x):
        return x[:3]
    else:
        return None


def first_four_letters_no_space(x):
    x = x.replace(' ', '')
    if len(x):
        return x[:4]
    else:
        return None


def first_five_letters_no_space(x):
    x = x.replace(' ', '')
    if len(x):
        return x[:5]
    else:
        return None


def sorted_integers(x):
    digits = re.compile(r'\d+').findall
    numeric_list = digits(x)
    numeric_list = sorted([int(n) for n in numeric_list])
    string_list = [str(n) for n in numeric_list]
    if len(string_list):
        return " ".join(string_list)
    else:
        return None


def first_integer(x):
    digits = re.compile(r'\d+').findall
    numeric_list = digits(x)
    if len(numeric_list):
        return numeric_list[0]
    else:
        return None


def last_integer(x):
    digits = re.compile(r'\d+').findall
    numeric_list = digits(x)
    if len(numeric_list):
        return numeric_list[-1]
    else:
        return None


def largest_integer(x):
    digits = re.compile(r'\d+').findall
    numeric_list = digits(x)
    numeric_list = sorted([int(n) for n in numeric_list])
    if len(numeric_list):
        return str(numeric_list[-1])
    else:
        return None


def three_letter_abbreviation(x):
    letters = re.compile((r'[a-zA-Z]+')).findall
    word_list = letters(x)
    if len(word_list) >= 3:
        abbreviation = "".join(w[0] for w in word_list[:3])
        return abbreviation
    else:
        return None


all_rules = [whole_field, first_word, first_two_words, first_three_letters, first_four_letters, first_five_letters,
             first_three_letters_no_space, first_four_letters_no_space, first_five_letters_no_space, sorted_integers,
             first_integer, last_integer, largest_integer, three_letter_abbreviation]
