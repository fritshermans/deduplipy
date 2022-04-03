from deduplipy.blocking.blocking_rules import (whole_field, first_word, first_two_words,
                                               first_three_letters, first_four_letters, first_five_letters,
                                               first_three_letters_no_space, first_four_letters_no_space,
                                               first_five_letters_no_space, sorted_integers, first_integer,
                                               last_integer, largest_integer, three_letter_abbreviation, last_word,
                                               last_two_words, last_three_letters, last_four_letters, last_five_letters,
                                               last_three_letters_no_space, last_four_letters_no_space,
                                               last_five_letters_no_space)


def test_whole_field():
    assert whole_field('one two 3') == 'one two 3'
    assert whole_field('one') == 'one'
    assert whole_field(' ') is None
    assert whole_field('') is None


def test_first_word():
    assert first_word('one two') == 'one'
    assert first_word(' one two') == 'one'
    assert first_word(' ') is None
    assert first_word('') is None


def test_last_word():
    assert last_word('one two') == 'two'
    assert last_word(' one two') == 'two'
    assert last_word(' ') is None
    assert last_word('') is None


def test_first_two_words():
    assert first_two_words('one two three') == 'one two'
    assert first_two_words('one two') == 'one two'
    assert first_two_words('one') == 'one'
    assert first_two_words(' one two three ') == 'one two'
    assert first_two_words(' ') is None
    assert first_two_words('') is None


def test_last_two_words():
    assert last_two_words('one two three') == 'two three'
    assert last_two_words('one two') == 'one two'
    assert last_two_words('one') == 'one'
    assert last_two_words(' one two three ') == 'two three'
    assert last_two_words(' ') is None
    assert last_two_words('') is None


def test_first_three_letters():
    assert first_three_letters('onetwo') == 'one'
    assert first_three_letters('one ') == 'one'
    assert first_three_letters(' one') == 'one'
    assert first_three_letters(' ') is None
    assert first_three_letters('') is None


def test_last_three_letters():
    assert last_three_letters('onetwo') == 'two'
    assert last_three_letters('one ') == 'one'
    assert last_three_letters(' one') == 'one'
    assert last_three_letters(' ') is None
    assert last_three_letters('') is None


def test_first_four_letters():
    assert first_four_letters('onetwo') == 'onet'
    assert first_four_letters('one ') == 'one'
    assert first_four_letters(' one') == 'one'
    assert first_four_letters('12345') == '1234'
    assert first_four_letters('12') == '12'
    assert first_four_letters(' ') is None
    assert first_four_letters('') is None


def test_last_four_letters():
    assert last_four_letters('onetwo') == 'etwo'
    assert last_four_letters('four ') == "four"
    assert last_four_letters(' four') == 'four'
    assert last_four_letters('12345') == '2345'
    assert last_four_letters('12') == '12'
    assert last_four_letters(' ') is None
    assert last_four_letters('') is None


def test_first_five_letters():
    assert first_five_letters('onetwo') == 'onetw'
    assert first_five_letters('one ') == 'one'
    assert first_five_letters(' one') == 'one'
    assert first_five_letters('123456') == '12345'
    assert first_five_letters('12') == '12'
    assert first_five_letters(' ') is None
    assert first_five_letters('') is None


def test_last_five_letters():
    assert last_five_letters('onetwo') == 'netwo'
    assert last_five_letters('one ') == 'one'
    assert last_five_letters(' one') == 'one'
    assert last_five_letters('123456') == '23456'
    assert last_five_letters('12') == '12'
    assert last_five_letters(' ') is None
    assert last_five_letters('') is None


def test_first_three_letters_no_space():
    assert first_three_letters_no_space('on etwo') == 'one'
    assert first_three_letters_no_space('one ') == 'one'
    assert first_three_letters_no_space(' one') == 'one'
    assert first_three_letters_no_space(' ') is None
    assert first_three_letters_no_space('') is None


def test_last_three_letters_no_space():
    assert last_three_letters_no_space('onet wo') == 'two'
    assert last_three_letters_no_space('one ') == 'one'
    assert last_three_letters_no_space(' one') == 'one'
    assert last_three_letters_no_space(' ') is None
    assert last_three_letters_no_space('') is None


def test_first_four_letters_no_space():
    assert first_four_letters_no_space('on e two') == 'onet'
    assert first_four_letters_no_space('one ') == 'one'
    assert first_four_letters_no_space(' one') == 'one'
    assert first_four_letters_no_space(' ') is None
    assert first_four_letters_no_space('') is None


def test_last_four_letters_no_space():
    assert last_four_letters_no_space('on e two') == 'etwo'
    assert last_four_letters_no_space('one ') == 'one'
    assert last_four_letters_no_space(' one') == 'one'
    assert last_four_letters_no_space(' ') is None
    assert last_four_letters_no_space('') is None


def test_first_five_letters_no_space():
    assert first_five_letters_no_space('on e two') == 'onetw'
    assert first_five_letters_no_space('one ') == 'one'
    assert first_five_letters_no_space(' one') == 'one'
    assert first_five_letters_no_space(' ') is None
    assert first_five_letters_no_space('') is None


def test_last_five_letters_no_space():
    assert last_five_letters_no_space('on e two') == 'netwo'
    assert last_five_letters_no_space('one ') == 'one'
    assert last_five_letters_no_space(' one') == 'one'
    assert last_five_letters_no_space(' ') is None
    assert last_five_letters_no_space('') is None


def test_sorted_integers():
    assert sorted_integers('2 1 word') == '1 2'
    assert sorted_integers('2 word 1') == '1 2'
    assert sorted_integers('2') == '2'
    assert sorted_integers('word') is None
    assert sorted_integers(' ') is None
    assert sorted_integers('') is None


def test_first_integer():
    assert first_integer('2 1 word') == '2'
    assert first_integer('word 2 1 word') == '2'
    assert first_integer('word2 1 word') == '2'
    assert first_integer('2') == '2'
    assert first_integer('word') is None
    assert first_integer(' ') is None
    assert first_integer('') is None


def test_last_integer():
    assert last_integer('2 1 word') == '1'
    assert last_integer('word 2 1 word') == '1'
    assert last_integer('word2 1 word') == '1'
    assert last_integer('2') == '2'
    assert last_integer('word') is None
    assert last_integer(' ') is None
    assert last_integer('') is None


def test_largest_integer():
    assert largest_integer('2 1 word') == '2'
    assert largest_integer('word 2 1 word') == '2'
    assert largest_integer('word2 1 wo99rd') == '99'
    assert largest_integer('2') == '2'
    assert largest_integer('word') is None
    assert largest_integer(' ') is None
    assert largest_integer('') is None


def test_three_letter_abbreviation():
    assert three_letter_abbreviation('one two three') == 'ott'
    assert three_letter_abbreviation('one two three four') == 'ott'
    assert three_letter_abbreviation('one.two three') == 'ott'
    assert three_letter_abbreviation('one two') is None
    assert three_letter_abbreviation('one') is None
    assert three_letter_abbreviation(' ') is None
    assert three_letter_abbreviation('') is None
