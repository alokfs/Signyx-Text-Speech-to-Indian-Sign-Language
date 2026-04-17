import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SIGN_FILES_DIR = BASE_DIR / "static" / "SignFiles"
WORDS_FILE = BASE_DIR / "words.txt"
SIGML_FILES_JSON = BASE_DIR / "static" / "js" / "sigmlFiles.json"


def build_sigml_index():
    sign_files = sorted(SIGN_FILES_DIR.glob("*.sigml"), key=lambda path: path.name.lower())
    return [
        {
            "fileName": sign_file.name,
            "name": sign_file.stem.lower(),
            "sid": index,
        }
        for index, sign_file in enumerate(sign_files, start=1)
    ]


def write_words_file(sign_index):
    with WORDS_FILE.open("w", encoding="utf-8") as words_file:
        for sign in sign_index:
            words_file.write(f"{sign['name']}\n")


def write_sigml_index(sign_index):
    with SIGML_FILES_JSON.open("w", encoding="utf-8") as sigml_file:
        json.dump(sign_index, sigml_file, indent=3)


def main():
    sign_index = build_sigml_index()
    write_words_file(sign_index)
    write_sigml_index(sign_index)


if __name__ == "__main__":
    main()
