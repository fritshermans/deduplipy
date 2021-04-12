import re


def whole_field(x):
    return x


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


def first_four_letters(x):
    if len(x):
        return x[:4]
    else:
        return ''


def first_five_letters(x):
    if len(x):
        return x[:5]
    else:
        return ''


def first_three_letters_no_space(x):
    x = x.replace(' ', '')
    if len(x):
        return x[:3]
    else:
        return ''


def first_four_letters_no_space(x):
    x = x.replace(' ', '')
    if len(x):
        return x[:4]
    else:
        return ''


def first_five_letters_no_space(x):
    x = x.replace(' ', '')
    if len(x):
        return x[:5]
    else:
        return ''


def sorted_integers(x):
    digits = re.compile(r'\d+').findall
    numeric_list = digits(x)
    numeric_list = sorted([int(n) for n in numeric_list])
    string_list = [str(n) for n in numeric_list]
    return " ".join(string_list)


def last_integer(x):
    digits = re.compile(r'\d+').findall
    numeric_list = digits(x)
    return numeric_list[-1]


def largest_integer(x):
    digits = re.compile(r'\d+').findall
    numeric_list = digits(x)
    numeric_list = sorted([int(n) for n in numeric_list])
    return str(numeric_list[-1])


def three_letter_abbreviation(x):
    letters = re.compile((r'[a-zA-Z]+')).findall
    word_list = letters(x)
    abbreviation = " ".join(w[0] for w in word_list)
    return abbreviation


all_rules = [whole_field, first_word, first_two_words, first_three_letters, first_four_letters, first_five_letters,
             first_three_letters_no_space, first_four_letters_no_space, first_five_letters_no_space, sorted_integers,
             last_integer, largest_integer, three_letter_abbreviation]
