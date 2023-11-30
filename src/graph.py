""" Graph building """
import os
import requests
from loguru import logger
from pyvis.network import Network

class GraphBuilder:
    """ Build visual graph for concept map """
    def __init__(self):
        self.dbr = 'http://dbpedia.org/resource/'
        self.dbd = 'http://dbpedia.org/data/'

        self.abstract = "http://dbpedia.org/ontology/abstract"
        self.depiction = "http://xmlns.com/foaf/0.1/depiction"

        # Graph Network params
        self.gravity = -2000
        self.central_gravity = 0.3
        self.spring_length = 250

    def get_entity_info(self, entity_url: str):
        """ Abstract+image for the entity URL """
        url = f"{entity_url.replace(self.dbr, self.dbd)}.json"
        try:
            response = requests.get(url, timeout=3600)
            if response.status_code == 200:
                data = response.json()
                if entity_url in data and self.abstract in data[entity_url] \
                        and self.depiction in data[entity_url]:
                    abstract = data[entity_url][self.abstract][0]["value"]
                    image = data[entity_url][self.depiction][0]["value"]
                    return abstract, image
            return None, None
        except Exception as e:
            logger.warning(e)
            return None, None

    def update_graph(self, graph: Network, lines: list[(str, str)]):
        """ Update graph with entities """
        for word, url in lines:
            abstract, image = self.get_entity_info(url)
            if abstract and image:
                graph.add_node(word, label=word, title=abstract, image=image, shape="image")
        return graph

    def init_graph(self):
        """ Init network graph for visualisation """
        graph = Network(height="750px", width="100%", directed=True)
        graph.barnes_hut(
            gravity=self.gravity, central_gravity=self.central_gravity, 
            spring_length=self.spring_length)
        return graph

    def main_folder(self, input_folder: str, output_folder: str):
        """ Main """
        for root, _, files in os.walk(input_folder):
            for file_name in [x for x in files if x.endswith(".txt")]:
                file_path = os.path.join(root, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    lines = file.readlines()
                    graph = self.init_graph()

                    lines = [x.strip().split(" - ") for x in lines]
                    lines = [(x[0], x[1]) for x in lines if len(x) >= 2]
                    graph = self.update_graph(graph=graph, lines=lines)

                    output_root = root.replace(input_folder, output_folder)
                    os.makedirs(output_root, exist_ok=True)

                    output_file_name = file_name.replace(".txt", "-entity_graph.html")
                    output_file_path = os.path.join(output_root, output_file_name)
                    graph.show_buttons(filter_=["physics"])
                    graph.write_html(output_file_path)
