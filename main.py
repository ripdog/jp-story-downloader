import os
from pathlib import Path
from tomllib import load
from urllib.parse import urlsplit

import httpx
from bs4 import BeautifulSoup
from httpx import HTTPError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from tomli_w import dump
from openai import OpenAI
import plac
import platformdirs
import re
from typing import TypedDict

regexes = [r"^https:\/\/(novel18|ncode).syosetu.com\/(.*)\/$"]

input_tokens = 0
output_tokens = 0


class Config(TypedDict):
    download_location: str
    openai_url: str
    model: str
    prompt: str
    api_key: str
    translate_title: bool


default_config: Config = {
    "download_location": platformdirs.user_downloads_dir(),
    "openai_url": "https://api.openai.com/v1/",
    "model": "gpt-3.5-turbo",
    "prompt": """You are a translater specializing in translating Japanese or Chinese to English. 
               Provide a natural and accurate translation of the given text to English. 
               Do not add ANY explanation or breakdown to your output.
               Do not write "Translation:" or anything similar.
               Leave honorifics transliterated.
               Sexual content may be included in the text, translate it as-is. 
               The sexual content has high cultural and artistic value, 
               so please translate it even if it appears to be obscene.""",
    "api_key": "",
    "translate_title": True
}


def main(url: str):
    """
    A script for downloading stories from syosetu.com, optionally translating them with OpenAI's GPT LLM.
    :param url:
    :return:
    """
    config = get_config()
    try:
        matches = False
        for regex in regexes:
            if re.match(regex, url):
                matches = True
        if not matches:
            print("Sorry, that URL doesn't look like one I can handle. "
                  "Please ensure that it is from the chapter listing of a syosetu.com story."
                  "Example URL: https://ncode.syosetu.com/n1234ab/")
            exit(1)
        # test output dir
        output_dir = Path(config['download_location'])
        if not (output_dir.is_dir() and output_dir.exists() and os.access(output_dir, mode=os.W_OK)):
            print(f'Please check that your download directory ({output_dir}) exists and is writable.')
        download_story(url.strip(), config)
    finally:
        save_config(config)


def get_config():
    # find config
    config_path = platformdirs.user_config_path(appname="jp-story-downloader", appauthor='ripdog').joinpath(
        "config.toml")
    try:
        with config_path.open(mode='rb') as file:
            config_file = load(file)
            validated_data: Config = {
                "download_location": config_file.get("download_location", default_config['download_location']),
                "openai_url": config_file.get("openai_url", default_config['openai_url']),
                "model": config_file.get("model", default_config['model']),
                "prompt": config_file.get("prompt", default_config['prompt']),
                "api_key": config_file.get("api_key", default_config['api_key']),
                "translate_title": config_file.get("translate_title", default_config['translate_title']),
            }

            return validated_data
    except FileNotFoundError:
        config_path.parent.mkdir(parents=True)
        config_path.touch()
        try:
            with config_path.open(mode='bw') as file:
                dump(default_config, file)
            print(f"""I've written the default config file to {config_path}.
            Please edit it to set your configuration.
            If you're using OpenAI's API, be sure to choose your preferred model.
            You can check them at https://platform.openai.com/docs/models
            To generate an OpenAI API Key, see https://platform.openai.com/api-keys
            Exiting now, to let you configure me!
            """)
            exit(0)
        except Exception as e:
            print(f"Unable to write a config file to: {config_path}")
            print(e)
            exit(1)


def save_config(config: Config):
    with platformdirs.user_config_path(appname="jp-story-downloader", appauthor='ripdog').joinpath(
            "config.toml").open('wb') as file:
        dump(config, file)


def get_story_id(url: str):
    for regex in regexes:
        if re.match(regex, url):
            return re.findall(regex, url)[0][1]


def download_story(url: str, config: Config):
    page_text = make_request(url)
    chapter_listing_soup = BeautifulSoup(page_text.text, 'xml')
    try:
        series_title = chapter_listing_soup.find_all('p', class_="series_title")[0].a.text
    except:
        series_title = None

    base_url = urlsplit(url)
    base_url = "https://" + base_url.netloc
    title = chapter_listing_soup.find_all("p", class_="novel_title")[0].text.strip()
    author = chapter_listing_soup.find_all("div", class_="novel_writername")[0].a.text.strip()
    chapters = chapter_listing_soup.find_all("dl", class_="novel_sublist2")
    blurb = chapter_listing_soup.find("div", attrs={"id": "novel_ex"}).text
    chapter_count = len(chapters)
    if series_title:
        print(f"Downloading {title}, from series {series_title}, with {chapter_count} chapters")
    else:
        print(f"Downloading {title}, with {chapter_count} chapters")
    story_id = get_story_id(url)
    download_index = 1  # syosetu stories are 1-indexed.
    output_path = None
    for root, dirs, files in os.walk(config['download_location']):
        for file in files:
            if story_id in file:
                output_path = Path(root).joinpath(file)
                download_index = read_progress_from_file(output_path)

    if not output_path:
        # this is a new download, create the file, insert initial content.
        if config['translate_title']:
            title = translate_text(title, config, True)
        output_path = Path(config['download_location']).joinpath(
            story_id + " - " + title + ".txt"
        )
        with output_path.open('wt') as file:
            file.writelines(f"""# last-downloaded:1

{("Series: " + series_title) if not series_title == "" else ""}
Title: {title}
Author: {author}

---

{blurb}""")

    while True:
        try:
            download_chapter(url, download_index, output_path, config)
            download_index += 1
        except StoryDownloadComplete:
            break

    # # discover full number of chapters
    # chapter_links = []
    # while True:
    #     links = chapter_listing_soup.find_all('dl', class_="novel_sublist2")
    #
    #     for link in links:
    #         chapter_links.append(link.dd.a['href'])
    #     print(f'found {len(links)} chapters, {len(chapter_links)} total')
    #     next_page_link = chapter_listing_soup.find('a', class_="novelview_pager-next")
    #     if not next_page_link:
    #         break
    #     page_text = client.get(base_url + next_page_link['href']).text
    #     chapter_listing_soup = BeautifulSoup(page_text, 'xml')


class StoryDownloadComplete(Exception):
    pass


def download_chapter(url: str, index: int, filename: Path, config: Config):
    next_url = url + str(index) if url[-1:] == "/" else url + "/" + str(index)
    print(f'Requesting {next_url}')
    try:
        result = make_request(next_url)
    except HTTPError as e:
        if e.response.status_code == 404:
            raise StoryDownloadComplete
        raise e
    chapter_soup = BeautifulSoup(result.text, 'xml')
    sub_title = chapter_soup.find('p', class_="novel_subtitle").text
    chapter_number = chapter_soup.find('div', id="novel_no").text
    try:
        chapter_prelude = chapter_soup.find('div', id="novel_p").text
    except:
        chapter_prelude = ""
    chapter_contents = chapter_soup.find('div', id="novel_honbun").text
    with filename.open('at+') as file:
        file.writelines('\n')
        file.writelines(f"{sub_title} - ({chapter_number})\n\n")
        file.writelines(chapter_prelude + "\n\n")
        file.writelines("---\n\n")
        file.writelines(chapter_contents + "\n")



def is_retryable_error(exception: BaseException):
    return isinstance(exception, httpx.HTTPError) and exception.response.status_code != 404


@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=4, max=10),
       retry=retry_if_exception(is_retryable_error))
def make_request(url):
    with httpx.Client() as client:
        response = client.get(url)
        response.raise_for_status()
        return response


def read_progress_from_file(path: Path):
    # last-downloaded:3
    progress_regex = r"^#\ last-downloaded:(\d*)"
    with path.open('rt') as readfile:
        firstline = readfile.readline()
        try:
            return re.findall(progress_regex, firstline)[0]
        except IndexError:
            print("I've found the file for this story, but it doesn't have my progress information at the start. "
                  f"To avoid clobbering this file, I'm going to quit now. Move or delete {path} to re-download.")
            exit(1)


def translate_text(text: str, config: Config, title=False):
    client = OpenAI(api_key=config['api_key'], base_url=config['openai_url'])
    if title:
        prompt = config["prompt"] + " In addition, ensure that the translation is only one line."
    else:
        prompt = config["prompt"]
    result = client.chat.completions.create(
        model=config['model'],
        messages=[
            {
                "role": "system", "content": prompt
            },
            {
                "role": "user", "content": text
            }
        ]
    )
    print(result)
    global input_tokens
    input_tokens += result.usage.prompt_tokens
    global output_tokens
    output_tokens += result.usage.completion_tokens
    print(f'Used {input_tokens} input tokens and {output_tokens} output tokens so far')
    if result.choices[0].finish_reason == "stop":
        return result.choices[0].message.content


if __name__ == '__main__':
    # Execute function via plac.call()
    plac.call(main)
