import os
from pyvis.network import Network


def generate_graph_from_txt_files(input_folder_path, output_folder_path):
    for file_name in os.listdir(input_folder_path):
        file_path = os.path.join(input_folder_path, file_name)

        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            with open(file_path, "r") as file:
                content = file.readlines()

            # Check if the file is empty
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

                # Skip lines with empty values
                if any(not element.strip() for element in elements):
                    continue

                source_node = elements[0].strip()
                edge_label = elements[1].strip()
                target_node = elements[2].strip()

                # if source_node and not g.get_node(source_node):
                g.add_node(source_node, label=source_node)

                # if target_node and not g.get_node(target_node):
                g.add_node(target_node, label=target_node)

                # if source_node and target_node:
                g.add_edge(source_node, target_node, label=edge_label)

            g.save_graph(graph_path)
            print(f"Graph generated for file: {file_name}")
