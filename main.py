import os
import ssl
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import stanza
from flask import Flask, jsonify, render_template, request, send_from_directory
from nltk.parse.stanford import StanfordParser
from nltk.tree import ParentedTree, Tree
from urllib import request as urllib_request

ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__, static_folder="static", static_url_path="")

BASE_DIR = Path(__file__).resolve().parent
STANFORD_PARSER_DIR = BASE_DIR / "stanford-parser-full-2018-10-17"
STANFORD_PARSER_ZIP = BASE_DIR / "stanford-parser-full-2018-10-17.jar"
STANFORD_MODELS_PATH = (
    STANFORD_PARSER_DIR
    / "edu"
    / "stanford"
    / "nlp"
    / "models"
    / "lexparser"
    / "englishPCFG.ser.gz"
)
STANZA_RESOURCES_DIR = BASE_DIR / "stanza_resources"
NLTK_DATA_DIR = BASE_DIR / "nltk_data"
WORDS_FILE = BASE_DIR / "words.txt"
SIGN_FILES_DIR = BASE_DIR / "static" / "SignFiles"
PARSER_DOWNLOAD_URL = "https://nlp.stanford.edu/software/stanford-parser-full-2018-10-17.zip"
STOP_WORDS = {
    "a",
    "am",
    "an",
    "and",
    "are",
    "is",
    "was",
    "were",
    "the",
    "be",
    "being",
    "been",
    "have",
    "has",
    "had",
    "does",
    "did",
    "could",
    "should",
    "would",
    "can",
    "but",
    "or",
    "shall",
    "will",
    "may",
    "might",
    "must",
    "let",
    "to",
}
SKIPPED_UPOS = {"AUX", "CCONJ", "DET", "SCONJ"}
SIGN_WORD_ALIASES = {
    "building": "build",
    "everyone": "every",
    "everybody": "every",
}
SIGN_PHRASE_ALIASES = {
    ("next", "year"): "nextyear",
    ("thank", "you"): "thankyou",
}

os.environ.setdefault("JAVAHOME", r"C:\Program Files\Java\jdk-17\bin\java.exe")
os.environ["CLASSPATH"] = str(STANFORD_PARSER_DIR)
os.environ["STANFORD_MODELS"] = str(STANFORD_MODELS_PATH)
os.environ["NLTK_DATA"] = str(NLTK_DATA_DIR)

start_time = 0.0
pipeline: Any | None = None
stanford_parser: StanfordParser | None = None
valid_words_cache: set[str] | None = None
valid_words_cache_key: tuple[int, int, int] | None = None


def get_pipeline() -> Any:
    global pipeline
    if pipeline is None:
        pipeline_kwargs = {
            "processors": {"tokenize": "spacy", "lemma": "combined_nocharlm"},
            "use_gpu": False,
        }
        if STANZA_RESOURCES_DIR.exists():
            pipeline_kwargs["model_dir"] = str(STANZA_RESOURCES_DIR)
            pipeline_kwargs["download_method"] = None
        pipeline = stanza.Pipeline("en", **pipeline_kwargs)
    return pipeline


def get_stanford_parser() -> StanfordParser:
    global stanford_parser
    if stanford_parser is None:
        stanford_parser = StanfordParser()
    return stanford_parser


def load_valid_words() -> set[str]:
    global valid_words_cache, valid_words_cache_key

    sign_files = list(SIGN_FILES_DIR.glob("*.sigml"))
    words_mtime = WORDS_FILE.stat().st_mtime_ns if WORDS_FILE.exists() else 0
    latest_sign_mtime = max((path.stat().st_mtime_ns for path in sign_files), default=0)
    cache_key = (words_mtime, len(sign_files), latest_sign_mtime)

    if valid_words_cache is None or valid_words_cache_key != cache_key:
        valid_words = set()

        if WORDS_FILE.exists():
            with WORDS_FILE.open("r", encoding="utf-8") as words_file:
                valid_words.update(
                    line.strip().lower() for line in words_file if line.strip()
                )

        valid_words.update(sign_file.stem.lower() for sign_file in sign_files)
        valid_words_cache = valid_words
        valid_words_cache_key = cache_key

    return valid_words_cache


def is_parser_jar_file_present():
    return STANFORD_PARSER_ZIP.exists()


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.perf_counter()
        return

    duration = time.perf_counter() - start_time
    if duration <= 0:
        return

    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write(
        "\r...%d%%, %d MB, %d KB/s, %d seconds passed"
        % (percent, progress_size / (1024 * 1024), speed, duration)
    )
    sys.stdout.flush()


def download_parser_jar_file():
    urllib_request.urlretrieve(
        PARSER_DOWNLOAD_URL, str(STANFORD_PARSER_ZIP), reporthook
    )


def extract_parser_jar_file():
    try:
        with zipfile.ZipFile(STANFORD_PARSER_ZIP) as zip_file:
            zip_file.extractall(path=BASE_DIR)
    except Exception:
        if STANFORD_PARSER_ZIP.exists():
            STANFORD_PARSER_ZIP.unlink()
        download_parser_jar_file()
        extract_parser_jar_file()


def extract_models_jar_file():
    models_jar = STANFORD_PARSER_DIR / "stanford-parser-3.9.2-models.jar"
    with zipfile.ZipFile(models_jar) as zip_file:
        zip_file.extractall(path=STANFORD_PARSER_DIR)


def download_required_packages():
    if not STANFORD_PARSER_DIR.exists():
        if not is_parser_jar_file_present():
            download_parser_jar_file()
        extract_parser_jar_file()

    if not STANFORD_MODELS_PATH.exists():
        extract_models_jar_file()


def normalize_input(text: str) -> str:
    cleaned_text = " ".join(text.strip().replace("\t", " ").split())
    if not cleaned_text:
        return ""
    if len(cleaned_text) == 1:
        return cleaned_text

    sentences = []
    for raw_sentence in cleaned_text.split("."):
        if clean_sentence := raw_sentence.strip():
            sentences.append(f"{clean_sentence.capitalize()} .")
    return " ".join(sentences)


def convert_to_word_list(sentences):
    word_list = []
    word_list_detailed = []

    for sentence in sentences:
        words = []
        detailed_words = []
        for word in sentence.words:
            words.append(word.text)
            detailed_words.append(word)
        word_list.append(words)
        word_list_detailed.append(detailed_words)

    return word_list, word_list_detailed


def align_reordered_details(reordered_words, detailed_words):
    remaining_details = list(detailed_words)
    aligned_details = []

    for token in reordered_words:
        match_index = next(
            (
                index
                for index, detail in enumerate(remaining_details)
                if detail.text == token or detail.text.lower() == token.lower()
            ),
            None,
        )

        if match_index is None:
            raise ValueError(f"Unable to align token '{token}' with stanza output")

        aligned_details.append(remaining_details.pop(match_index))

    return aligned_details


def remove_punct(words, detailed_words):
    clean_words = []
    clean_details = []

    for word, detail in zip(words, detailed_words):
        if detail.upos != "PUNCT":
            clean_words.append(word)
            clean_details.append(detail)

    return clean_words, clean_details


def filter_words(words, detailed_words):
    filtered_words = []
    filtered_details = []

    for word, detail in zip(words, detailed_words):
        normalized_word = word.lower()
        if normalized_word in STOP_WORDS or detail.upos in SKIPPED_UPOS:
            continue

        filtered_words.append(word)
        filtered_details.append(detail)

    return filtered_words, filtered_details


def lemmatize(words, detailed_words):
    lemmatized_words = []

    for word, detail in zip(words, detailed_words):
        if len(word) == 1:
            lemmatized_words.append(word)
        else:
            lemmatized_words.append(detail.lemma or word)

    return lemmatized_words


def label_parse_subtrees(parent_tree):
    return {sub_tree.treeposition(): 0 for sub_tree in parent_tree.subtrees()}


def handle_noun_clause(index, tree_traversal_flag, modified_parse_tree, sub_tree):
    parent_position = sub_tree.parent().treeposition()
    subtree_position = sub_tree.treeposition()

    if tree_traversal_flag[subtree_position] == 0 and tree_traversal_flag[parent_position] == 0:
        tree_traversal_flag[subtree_position] = 1
        modified_parse_tree.insert(index, sub_tree)
        index += 1

    return index, modified_parse_tree


def handle_verb_prop_clause(index, tree_traversal_flag, modified_parse_tree, sub_tree):
    for child_sub_tree in sub_tree.subtrees():
        if child_sub_tree.label() not in {"NP", "PRP"}:
            continue

        parent_position = child_sub_tree.parent().treeposition()
        subtree_position = child_sub_tree.treeposition()
        if tree_traversal_flag[subtree_position] == 0 and tree_traversal_flag[parent_position] == 0:
            tree_traversal_flag[subtree_position] = 1
            modified_parse_tree.insert(index, child_sub_tree)
            index += 1

    return index, modified_parse_tree


def modify_tree_structure(parent_tree):
    tree_traversal_flag = label_parse_subtrees(parent_tree)
    modified_parse_tree = Tree("ROOT", [])
    index = 0

    for sub_tree in parent_tree.subtrees():
        if sub_tree.label() == "NP":
            index, modified_parse_tree = handle_noun_clause(
                index, tree_traversal_flag, modified_parse_tree, sub_tree
            )
        if sub_tree.label() in {"VP", "PRP"}:
            index, modified_parse_tree = handle_verb_prop_clause(
                index, tree_traversal_flag, modified_parse_tree, sub_tree
            )

    for sub_tree in parent_tree.subtrees():
        for child_sub_tree in sub_tree.subtrees():
            parent_position = child_sub_tree.parent().treeposition()
            subtree_position = child_sub_tree.treeposition()
            if (
                len(child_sub_tree.leaves()) == 1
                and tree_traversal_flag[subtree_position] == 0
                and tree_traversal_flag[parent_position] == 0
            ):
                tree_traversal_flag[subtree_position] = 1
                modified_parse_tree.insert(index, child_sub_tree)
                index += 1

    return modified_parse_tree


def reorder_eng_to_isl(input_tokens):
    if all(len(word) == 1 for word in input_tokens):
        return input_tokens

    try:
        return _extracted_from_reorder_eng_to_isl_6(input_tokens)
    except Exception as exc:
        print(f"Falling back to original word order: {exc}")
        return input_tokens


# TODO Rename this here and in `reorder_eng_to_isl`
def _extracted_from_reorder_eng_to_isl_6(input_tokens):
    parser: StanfordParser = get_stanford_parser()
    possible_parse_tree_list = list(parser.parse(input_tokens))
    if not possible_parse_tree_list:
        return input_tokens

    parse_tree = possible_parse_tree_list[0]
    parent_tree = ParentedTree.convert(parse_tree)
    modified_parse_tree = modify_tree_structure(parent_tree)
    return modified_parse_tree.leaves()


def preprocess_sentence(words, detailed_words):
    words_without_punct, details_without_punct = remove_punct(words, detailed_words)
    filtered_words, filtered_details = filter_words(
        words_without_punct, details_without_punct
    )
    return lemmatize(filtered_words, filtered_details)


def apply_sign_aliases(words):
    aliased_words = []
    index = 0

    while index < len(words):
        if index + 1 < len(words):
            phrase_alias = SIGN_PHRASE_ALIASES.get(
                (words[index].lower(), words[index + 1].lower())
            )
            if phrase_alias:
                aliased_words.append(phrase_alias)
                index += 2
                continue

        normalized_word = words[index].lower()
        aliased_words.append(SIGN_WORD_ALIASES.get(normalized_word, normalized_word))
        index += 1

    return aliased_words


def final_output(words):
    valid_words = load_valid_words()
    final_words = []

    for normalized_word in apply_sign_aliases(words):
        if normalized_word in valid_words:
            final_words.append(normalized_word)
        else:
            final_words.extend(normalized_word)

    return final_words


def convert(doc):
    download_required_packages()
    final_output_in_sent = []
    word_list, word_list_detailed = convert_to_word_list(doc.sentences)

    for words, detailed_words in zip(word_list, word_list_detailed):
        reordered_words = reorder_eng_to_isl(words)
        try:
            reordered_details = align_reordered_details(reordered_words, detailed_words)
        except ValueError:
            reordered_words = words
            reordered_details = detailed_words

        processed_words = preprocess_sentence(reordered_words, reordered_details)
        final_output_in_sent.append(final_output(processed_words))

    return final_output_in_sent


def take_input(text):
    normalized_text = normalize_input(text)
    if not normalized_text:
        return []

    document = get_pipeline()(normalized_text)
    return convert(document)


def build_response_words(final_output_in_sent):
    final_words_dict = {}
    word_index = 1

    for words in final_output_in_sent:
        for word in words:
            final_words_dict[word_index] = word.upper() if len(word) == 1 else word
            word_index += 1

    return final_words_dict


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    text = (request.form.get("text") or "").strip()
    if not text:
        return jsonify({})

    try:
        translated_words = take_input(text)
        return jsonify(build_response_words(translated_words))
    except Exception:
        app.logger.exception("Translation failed")
        return jsonify({"error": "Translation failed"}), 500


@app.route("/static/<path:path>")
def serve_signfiles(path):
    return send_from_directory("static", path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
