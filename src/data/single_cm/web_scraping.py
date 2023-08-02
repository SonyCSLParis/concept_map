import os
import re

import requests
from bs4 import BeautifulSoup

main_url = "http://web.archive.org/web/20030210113823/http://www.biologylessons.sdsu.edu/ta/toc.html"

file_counter = 1


def fetch_content_before_exercise(page_url):
    global file_counter
    response = requests.get(page_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        exercise_heading = soup.find("h2", text=re.compile(r"Exercise\s+1", re.IGNORECASE))
        if exercise_heading:
            td_parent = exercise_heading.find_parent("td")
            if td_parent:
                content_before_exercise = ""
                for sibling in td_parent.find_all_previous(recursive=False):
                    # if sibling.name == "td":
                    #     break
                    content_before_exercise = sibling.get_text() + content_before_exercise
                file_name = f"lesson_{file_counter}.txt"
                file_counter += 1
                return content_before_exercise, file_name
            else:
                print(f"Parent <td> not found for 'Exercise 1' in {page_url}")
        else:
            print(f"'Exercise 1' not found in {page_url}")
    else:
        print(f"Failed to fetch {page_url}: Status code {response.status_code}")
    return None, None

def fetch_content_before_heading(page_url, target_heading):
    global file_counter
    response = requests.get(page_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        target_heading_element = soup.find("b", text=re.compile(target_heading, re.IGNORECASE))
        if target_heading_element:
            # Find the content before the target heading
            content_before_heading = ""
            for sibling in target_heading_element.find_all_previous(recursive=False):
                if sibling.name == 'table':
                    break
                content_before_heading = sibling.get_text() + content_before_heading

            # Generate a unique filename and update the counter
            file_name = f"lesson_{file_counter}.txt"
            file_counter += 1
            return content_before_heading.strip(), file_name
        else:
            print(f"Target heading not found in {page_url}")
    else:
        print(f"Failed to fetch {page_url}: Status code {response.status_code}")
    return None, None

def fetch_content_before_heading_a(page_url, target_heading):
    global file_counter
    response = requests.get(page_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        # Finding the target heading element inside <b> tag
        target_heading_element = soup.find("b", text=re.compile(target_heading, re.IGNORECASE))
        if target_heading_element:
            # Find the parent table that contains the target heading
            parent_table = target_heading_element.find_parent("table")
            if parent_table:
                # Collect the content before the target heading
                content_before_heading = ""
                for row in parent_table.find_all_previous("tr", recursive=False):
                    if row.find("b", text=re.compile(target_heading, re.IGNORECASE)):
                        break
                    content_before_heading = row.get_text() + content_before_heading

                # Generate a unique filename and update the counter
                file_name = f"lesson_{file_counter}.txt"
                file_counter += 1
                return content_before_heading.strip(), file_name
            else:
                print(f"Parent <table> not found for {target_heading} in {page_url}")
        else:
            print(f"Target heading not found in {page_url}")
    else:
        print(f"Failed to fetch {page_url}: Status code {response.status_code}")
    return None, None

def main_scraping():
    output_directory = "content_before_exercise_files"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    response = requests.get(main_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        lesson_links = soup.find_all("a", text="A. Biology Lesson")
        for link in lesson_links:
            lesson_name = link.get_text().strip()
            lesson_url = link["href"]
            content_before_exercise, file_name = fetch_content_before_exercise(lesson_url)
            # if content_before_exercise == None :
            #     content_before_exercise, file_name = fetch_content_before_heading_a(lesson_url, "Exercise 1:")
            #     if content_before_exercise == None :
            #         content_before_exercise, file_name = fetch_content_before_heading(lesson_url, "Exercise 1:")

        if content_before_exercise and file_name:
            filename = os.path.join(output_directory, file_name)
            with open(filename, "w", encoding="utf-8") as file:
                file.write(content_before_exercise)
            print(f"Saved {lesson_name}.txt")
    else:
        print(f"Failed to fetch {main_url}: Status code {response.status_code}")


if __name__ == '__main__':
    main_scraping()
