import re


def first_word(x):
    if len(x):
        return x.split()[0]
    else:
        return ''


def first_two_words(x):
    if len(x):
        return " ".join(x.split()[:2])
    else:
        return ''


def first_three_letters(x):
    if len(x):
        return x[:3]
    else:
        return ''


def sorted_integers(x):
    digits = re.compile(r'\d+').findall
    numeric_list = sorted(digits(x))
    return " ".join(numeric_list)


def three_letter_abbreviation(x):
    letters = re.compile((r'[a-zA-Z]+')).findall
    word_list = letters(x)
    abbreviation = " ".join(w[0] for w in word_list)
    return abbreviation


all_rules = [first_word, first_two_words, first_three_letters, sorted_integers, three_letter_abbreviation]
