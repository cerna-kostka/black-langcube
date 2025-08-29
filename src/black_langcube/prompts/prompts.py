translator_eng = {
    "system": "You are a translator. Translate the input from {language} "
               "to English if not already written in English."
}

translator_usr = {
    "system": "You are a translator. Translate the input from English to {language}."
}

relevance = {
    "human": "Is article called '{title}' relevant for topic '{topic}' ? Answer Yes/No only:"
}

is_language = {
    "system": "Identify the language of the provided text. "
               "Respond with the English name of the language."
}

theme_analyzer = {
    "system": "foo"
}

keywords = {
    "system": "Bar",
    "human": "Baz"
}

strategy = {
    "system": "consectetur",
    "human": "adipiscing elit"
}

article_analyst = {
    "system": "nomen est omen",
}

outline = {
    "system": "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
                "The output must not contain any explanatory text or comments. Output just the outline itself in JSON format structured according to {outline_format}."
}

text = {
    "system": "Text. STRICTLY ADHERE to the described format:{chapter_format}"
}

check_title = {
    "system": "Identify language of the following input. Answer only with the English name of the language.",
    "generate_translation_system": "Translate the following text into the specified language."
}

title_abstract = {
    "system": "dolor sit"           
}