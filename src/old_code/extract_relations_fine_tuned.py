import regex as re

from fine_tuning_rebel.run_rebel import *
from settings import *


def extract_entity(text):
    match = re.match(r'(.+?) - ', text)
    if match:
        return match.group(1)
    return None


def extract_parent_folder(path):
    parent_folder = os.path.basename(os.path.dirname(path))
    return parent_folder


def preprocess_file(file_path):
    base_name = os.path.basename(file_path)
    prefix = ''.join(filter(str.isdigit, base_name))
    new_file_name = f"M{prefix}-preprocessed.txt"
    new_file_path = file_path.replace(base_name, new_file_name)
    return new_file_path


def entity_extracted_rebel(entity_path, text_path, output_path):
    for root, dirs, files in os.walk(entity_path):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)
                name_of_file = preprocess_file(file_name)
                parent_folder_name = extract_parent_folder(file_path)
                with open(file_path, "r") as file:
                    entities = []
                    for line in file:
                        parts = line.strip().split(' - ')
                        if len(parts) == 2:
                            entity_name, entity_url = parts
                            entities.append(entity_name)

                tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
                model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
                # model = torch.load("src/fine_tune_rebel/finetuned_rebel.pth")
                gen_kwargs = {
                    "max_length": 256,
                    "length_penalty": 0,
                    "num_beams": 3,
                    "num_return_sequences": 3,
                }
                new_path = os.path.join(text_path, parent_folder_name, name_of_file)
                with open(new_path, 'r') as file:
                    file_content = file.readlines()
                    output_file_path = os.path.join(output_path, parent_folder_name)
                    os.makedirs(output_file_path, exist_ok=True)

                    output_file_name = file_name.replace("-preprocessed-entities.txt", "-fine-tuned-relations.txt")
                    output_file_path = os.path.join(output_file_path, output_file_name)
                    with open(output_file_path, "w") as output_file:
                        for sentence in file_content:
                            model_inputs = tokenizer(sentence, max_length=256, padding=True, truncation=True,
                                                     return_tensors='pt')

                            generated_tokens = model.generate(
                                model_inputs["input_ids"].to(model.device),
                                attention_mask=model_inputs["attention_mask"].to(model.device),
                                **gen_kwargs,
                            )

                            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

                            # Extract triplets
                            # test_entities = ['James madison']
                            for entity in entities :
                                for idx, sentence in enumerate(decoded_preds):
                                    if entity in sentence:
                                        print(extract_triplets(sentence))
                                        output_file.write(str(extract_triplets(sentence)))
                                else:
                                    print('no triplets found')


if __name__ == '__main__':
    entity_path = OUTPUT_EXTRACTION_ENTITY
    text_path = OUTPUT_PREPROCESSING
    output_path = OUTPUT_TRIPLETS_FINE_TUNE
    entity_extracted_rebel(entity_path, text_path, output_path)
