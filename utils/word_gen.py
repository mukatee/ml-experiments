import random

NUM_WORDS = 50
MIN_LENGTH = 5
MAX_LENGTH = 10

vowels = "aeiouy"
consonants = "bcdfghjklmnpqrstvwxyz"
numbers = "0123456789"

for x in range(NUM_WORDS):
    world_length = random.randint(MIN_LENGTH, MAX_LENGTH)
    word = ""#"bitg"
    for w in range(world_length):
        prob_vowel = 50
        prob_num = 10
        if word[-1:] in vowels:
            prob_vowel = 30
            if word[-2:-1] in vowels:
                prob_vowel = 10
        if word[-1:] in consonants:
            prob_vowel += 10
            if word[-2:-1] in consonants:
                prob_vowel += 10
        prob = random.randint(0, 100)
        if prob <= prob_vowel:
            letter = random.choice(vowels)
        elif prob <= prob_vowel + prob_num:
            letter = random.choice(numbers)
        else:
            letter = random.choice(consonants)
        word += letter
    #word += "z"
    print(word)
