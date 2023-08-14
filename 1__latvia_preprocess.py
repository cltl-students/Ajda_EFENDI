# This code is the webscraping code for Latvian dataset
path='Latvia.txt'
from bs4 import BeautifulSoup
import csv
def clean_data(xml_data):
    soup = BeautifulSoup(xml_data, 'xml')

    qualifications = soup.find_all('tns:Qualification')

    cleaned_data = []

    for qualification in qualifications:
        title = qualification.find('tns:Title', language='en').text
        descriptions = qualification.find_all('tns:Description', language='en')
        level = qualification.find('tns:EQFLevel').text

        cleaned_descriptions = []
        for description in descriptions:
            description_text = description.get_text()
            cleaned_description_text = clean_description(description_text)
            cleaned_descriptions.append(cleaned_description_text)

        cleaned_qualification = {
            'title': title,
            'eqf_level_id': level,
            'description': cleaned_descriptions
        }

        cleaned_data.append(cleaned_qualification)

    return cleaned_data


def clean_description(description_text):
    soup = BeautifulSoup(description_text, 'html.parser')
    cleaned_text = soup.get_text()
    return cleaned_text

def main():
    # Read XML data from a file or any other source
    with open(path, 'r') as file:
        xml_data = file.read()

    cleaned_data = clean_data(xml_data)
    # # Open the output file in write mode
    with open('latvian_dataset', 'w') as file:
        # Write the headers
        writer = csv.writer(file)
        writer.writerow(['title', 'description', 'eqf_level_id'])
        # Write the data
        for qualification in cleaned_data:
            title = qualification['title']
            descriptions = qualification['description']
            level = qualification['eqf_level_id']
            # Write each description in a separate row
            for description in descriptions:
                writer.writerow([title, description, level])

    with open('latvian_dataset', 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        i = {name: index for index, name in enumerate(headers)}
        for line in reader:
            print(line[i['title']], line[i['eqf_level_id']])

# Call the main function
main()