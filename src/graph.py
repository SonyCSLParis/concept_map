import requests
from pyvis.network import Network
from settings import *

def get_entity_info(entity_url):
    # Fetching abstract and image for the entity URL
    url = f"{entity_url.replace('http://dbpedia.org/resource/', 'http://dbpedia.org/data/')}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if entity_url in data and "http://dbpedia.org/ontology/abstract" in data[entity_url] \
                and "http://xmlns.com/foaf/0.1/depiction" in data[entity_url]:
            abstract = data[entity_url]["http://dbpedia.org/ontology/abstract"][0]["value"]
            image = data[entity_url]["http://xmlns.com/foaf/0.1/depiction"][0]["value"]
            return abstract, image
    return None, None

def create_graph(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)
                with open(file_path, "r") as file:
                    lines = file.readlines()
                    graph = Network(height="750px", width="100%", directed=True)
                    graph.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=250)

                    for line in lines:
                        parts = line.strip().split(" - ")
                        if len(parts) >= 2:
                            word = parts[0]
                            url = parts[1]
                            abstract, image = get_entity_info(url)
                            if abstract and image:
                                graph.add_node(word, label=word, title=abstract, image=image, shape="image")

                output_root = root.replace(input_folder, output_folder)
                os.makedirs(output_root, exist_ok=True)

                output_file_name = file_name.replace(".txt", "-entity_graph.html")
                output_file_path = os.path.join(output_root, output_file_name)

                with open(output_file_path, "w") as output_file:
                    graph.show_buttons(filter_=["physics"])
                    graph.write_html(output_file_path)


