import os

import requests
from bs4 import BeautifulSoup
import re

main_url = "http://web.archive.org/web/20030210113823/http://www.biologylessons.sdsu.edu/ta/toc.html"

file_counter = 1

def fetch_content_before_exercise(page_url):
    global file_counter
    response = requests.get(page_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        # Finding the <h2> element containing "Exercise 1"
        exercise_heading = soup.find("h2", text=re.compile(r"Exercise\s+1", re.IGNORECASE))
        if exercise_heading:
            # Finding the parent <td> element of the exercise heading
            td_parent = exercise_heading.find_parent("td")
            if td_parent:
                # Getting all content before the exercise heading
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
        # Finding the target heading element
        target_heading_element = soup.find("a", text=re.compile(target_heading, re.IGNORECASE))
        if target_heading_element:
            # Find the second <td> element inside the second <tr> after the target heading
            td_elements = soup.find_all("td")
            found_target_heading = False
            for td in td_elements:
                if found_target_heading:
                    content_before_heading = td.get_text().strip()
                    break
                if target_heading in td.get_text().strip():
                    found_target_heading = True
            else:
                print(f"Target heading not found in {page_url}")
                return None, None

            # Generate a unique filename and update the counter
            file_name = f"lesson_{file_counter}.txt"
            file_counter += 1
            return content_before_heading, file_name
        else:
            print(f"Target heading not found in {page_url}")
    else:
        print(f"Failed to fetch {page_url}: Status code {response.status_code}")
    return None, None

def main_scraping():
    # Create a directory to store the TXT files
    output_directory = "content_before_exercise_files"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Fetch the main page
    response = requests.get(main_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        # Finding all links with the text "A. Biology Lesson"
        lesson_links = soup.find_all("a", text="A. Biology Lesson")
        for link in lesson_links:
            lesson_name = link.get_text().strip()
            lesson_url = link["href"]
            target_heading_element = soup.find("h2", text="How Does a Green Plant Grow?")
            if target_heading_element :
                target_heading = "How Does a Green Plant Grow?"
                content_before_exercise,file_name = fetch_content_before_heading(lesson_url, target_heading)
            else:
                content_before_exercise, file_name = fetch_content_before_exercise(lesson_url)
        if content_before_exercise and file_name:
                # Save the content before exercise to a TXT file
                filename = os.path.join(output_directory, file_name)
                with open(filename, "w", encoding="utf-8") as file:
                    file.write(content_before_exercise)
                print(f"Saved {lesson_name}.txt")
    else:
        print(f"Failed to fetch {main_url}: Status code {response.status_code}")

if __name__ == '__main__':
    main_scraping()
