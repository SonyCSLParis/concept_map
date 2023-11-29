import os
from pyvis.network import Network
import requests
import json

url = "https://symbotalkapiv1.azurewebsites.net/search/"
repo ="arassac"

def find_picto(x, url, repo):
    querystring = {"name": x, "lang": "en", "repo": repo, "limit": "1"}
    response = requests.request("GET", url, params=querystring)
    print(response)
    try:
        data = response.json()
    except json.JSONDecodeError:
        print("Error occurred while decoding the JSON response.")
        return None

    if response.status_code == 200:
        pictograms = []
        for item in data:
            lemma = item.get('lemma')
            pictogram_url = item.get('image_url')
            pictograms.append((lemma, pictogram_url))

        return pictograms
    else:
        print("Error occurred while querying the API.")
        return None

def generate_graph_from_txt_files(input_folder_path, output_folder_path):
    for file_name in os.listdir(input_folder_path):
        file_path = os.path.join(input_folder_path, file_name)

        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            with open(file_path, "r") as file:
                content = file.readlines()

            if len(content) == 0:
                print(f"Skipping empty file: {file_name}")
                continue

            graph_title = os.path.splitext(file_name)[0]
            graph_name = f"{graph_title}-graph.html"
            graph_path = os.path.join(output_folder_path, graph_name)

            g = Network()
            g.force_atlas_2based()
            g.show_buttons()

            for line in content:
                line = line.strip()
                elements = line.split(",")

                if any(not element.strip() for element in elements):
                    continue

                source_node = elements[0].strip()
                edge_label = elements[1].strip()
                target_node = elements[2].strip()


                # Call find_picto function to get image URL for source and target nodes
                source_node_url = find_picto(source_node, url, repo)
                target_node_url = find_picto(target_node, url, repo)

                if type(source_node_url) == 'list':
                    source_url = source_node_url[0][1]
                    print(type(source_url))
                    print(source_url)
                    g.add_node(source_node, label=source_node,shape='image',image=source_url)
                else :
                    g.add_node(source_node, label=source_node)

                if type(target_node_url) == 'list':
                    target_url = target_node_url[0][1]
                    print(type(target_url))
                    print(target_url)
                    g.add_node(target_node, label=target_node, shape='image', image=target_url)
                else :
                    g.add_node(target_node, label=target_node)

                g.add_edge(source_node, target_node, label=edge_label)

            g.save_graph(graph_path)
            print(f"Graph generated for file: {file_name}")
