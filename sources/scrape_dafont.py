from bs4 import BeautifulSoup
import urllib.request
import zipfile
import requests
import time
import os

# Need to submit a user agent
# https://stackoverflow.com/questions/25491872/request-geturl-returns-empty-content
ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.92 Safari/537.36"
url_format = "http://www.dafont.com/alpha.php?lettre=%s&page=%d"

total_count = 0
download_count = 0
for letter in range(ord('a'), ord('z') + 1):
    letter = chr(letter)

    page_count = 1
    current_page = 0

    while current_page < page_count:
        current_page += 1 # Start at page 1
        output_dir = f"{letter}/{current_page:03d}"
        os.makedirs(output_dir, exist_ok=True)
        existing_zips = set([f for f in os.listdir(output_dir) if f.endswith(".zip")])

        url = url_format % (letter, current_page)
        #  url = url_format % ("%23", current_page)

        try:
            resp = requests.get(url, headers = {"User-Agent" : ua})
            if not resp.ok:
                continue

            parsed = BeautifulSoup(resp.text, "lxml")

            # Update the page count
            noindex = parsed.find_all("div", class_="noindex")
            page_count = int(list(noindex[0].children)[-2].text.strip())
            print(f"Parsing page {current_page} of {page_count} from {url} ...")

            # https://stackoverflow.com/questions/5041008/how-to-find-elements-by-class
            links = parsed.find_all("a", class_ = "dl")
            for i, link in enumerate(links):
                total_count += 1

                location = link.get("href")
                font_name = location[location.rfind("=") + 1:]
                zip_name = f"{font_name}.zip"
                if zip_name in existing_zips:
                    print(f"[Font {total_count}] - Skipping download of {zip_name}!")
                    continue

                download_count += 1
                print(f"[Font {total_count}] - Downloading font {font_name} ({download_count} downloaded)")

                output_zip_name = f"{output_dir}/{zip_name}"
                urllib.request.urlretrieve("http:%s" % location, output_zip_name)

                # https://stackoverflow.com/questions/3451111/unzipping-files-in-python
                with zipfile.ZipFile(output_zip_name, "r") as zip_ref:
                    zip_ref.extractall(f"{output_dir}/{font_name}")

                # Let's not hammer the server ...
                time.sleep(1)
        except Exception as e:
            print(e)
            with open("errors.log", "a") as f:
                f.write(f"Failed to download from URL {url}: {str(e)}\n")

        # Let's not hammer the server ...
        time.sleep(1)
