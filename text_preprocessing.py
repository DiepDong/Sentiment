import re
import emoji

emoticon_to_word = {
    "🙂": "happy",
    "🙂": "happy",
    "😊": "happy",
    ":v": "happy",
    "😃": "laugh",
    "🙁": "sad",
    "😉": "wink",
    "😐": "neutral",
    "😛": "tongue",
    "😢": "cry",
    "😮": "surprise",  
    "^^": "happy"
}
english_to_vietnamese = {
    "happy": "vui vẻ ",
    "laugh": "cười ",
    "sad": "buồn ",
    "wink": "nháy mắt ",
    "neutral": "trung lập ",
    "tongue": "nói năng ",
    "cry": "khóc ",
    "surprise": "ngạc nhiên "
}
translate_emoticons_to_Vietnamese = {
    ":pouting_face:": " tức giận",
    ":face_with_tears_of_joy:": "haha ",
    ":beaming_face_with_smiling_eyes:": "vui nhỉ ",
    ":smiling_face_with_smiling_eyes:": "vui ghê ",
    ":loudly_crying_face:": "buồn quá ",
    ":expressionless_face:": "cạn lời ",
    ":hot_face:": "mệt thật ",
    ":rolling_on_the_floor_laughing:": "vui quá ",
    ":smiling_face_with_sunglasses:": "chất đó ",
    ":face_blowing_a_kiss:": "đáng yêu ",
    ":face_with_rolling_eyes:": "kinh ",
    ":grinning_face_with_sweat:": "ngại nhỉ ",
    ":grinning_face:": "hớn hở ",
    ":smiling_face_with_heart-eyes:": "đáng yêu ",
    ":red_heart:": "yêu yêu ",
    ":slightly_frowning_face:": "buồn quá ",
    ":pensive_face:": "chán nhỉ ",
    ":weary_face:": "mệt mỏi ",
    ":face_with_raised_eyebrow:": "khó nhỉ ",
    ":grinning_squinting_face:": "haha ",
    ":sleepy_face:": "thôi mệt ",
    ":crying_face:": "huhu buồn quá ",
    "surpriseK_hand:": "tốt ",
    ":smirking_face:": "khinh bỉ ",
    ":relieved_face:":"hihi ",
    ":face_vomiting:": "kinh tởm ",
    ":smiling_face_with_horns:":"kinh tởm",
}

def remove_repeated_sequences(text):
    # Define a regex pattern to find sequences of repeated characters (including punctuation)
    pattern = r'(\S)\1+'
    
    # Replace the repeated sequences with a single instance of the character
    text = re.sub(pattern, r'\1', text, flags=re.IGNORECASE)
    
    return text


def transform_emoticons(text):
    for emoticon, word in emoticon_to_word.items():
        text = text.replace(emoticon, word)
    return text

def transform_emojis_and_emoticons(text):
    text_with_emojis = emoji.demojize(text)
    text = transform_emoticons(text_with_emojis)
    return text

def translate_english_to_vietnamese(text):
    for english, vietnamese in english_to_vietnamese.items():
        text = text.replace(english, vietnamese)
    return text
def translate_english_to_vietnamese_emoticon(text):
    for word_form, vietnamese in translate_emoticons_to_Vietnamese.items():
        text = text.replace(word_form, vietnamese)
    return text
def replace_words(text):
    correct_words = {
        "cóa": "có",
        "coá": "có",
        "ngta": "người ta",
        "nta": "người ta",
        "cf": "cà phê",
        "coffee": "cà phê",
        "cafe": "cà phê ",
        "caphe": "cà phê",
        "coffe": "cà phê",
        "hk": "không",
        "cũnh": "cũng",
        "cungc": "cũng",
        "cungz": "cũng",
        "pik": "biết",
        "pk": "biết",
        "bik": "biết",
        "bjt": "biết",
        "nge": "nghe",
        "t": "tui",
        "uh": "ừm",
        "uhm": "ừm",
        "ah": "à",
        "v": "vậy",
        "thậc": "thật",
        "e":"em",
        "a": "anh",
        }
    words = text.split()
    replaced_words = [correct_words.get(word, word) for word in words]
    text = ' '.join(replaced_words)
    return text

def remove_stopwords(text, stopword_file='stopword.txt'):
    # Read stopwords from the file with utf-8 encoding
    with open(stopword_file, 'r', encoding='utf-8') as file:
        stopwords = set(word.strip() for word in file)

    # Tokenize the input sentence into words
    words = text.split()

    # Remove the stopwords from the list of words
    filtered_words = [word for word in words if word.lower() not in stopwords]

    # Reconstruct the sentence without the stopwords
    text = ' '.join(filtered_words)

    return text

def process_text(input_text):
    # Applying all transformations in order
    text = remove_repeated_sequences(input_text)
    text = transform_emojis_and_emoticons(text)
    text = translate_english_to_vietnamese_emoticon(text)
    text = translate_english_to_vietnamese(text)
    text = replace_words(text)
    text = remove_stopwords(text)

    return text