import os
import json
import hashlib 
from datasets import load_dataset
from tqdm import tqdm

FOLDER_NAME = os.path.dirname(os.path.abspath(__file__))

TASK_LIST= ['CodeTrans', 'CodeSearchNet', 'BFP', 'CONCODE']
TEXT_KEYS = {'CONCODE': 'nl',
            'CodeTrans': 'java',
            'CodeSearchNet': 'code',   
            'BFP': 'buggy'}
LABEL_KEYS = {'CONCODE': 'code',
            'CodeTrans': 'cs',
            'CodeSearchNet': 'docstring',
            'BFP': 'fixed'}

DEFINITION ={ 'CONCODE': 'Generate Java code from the following English description: ',
            'CodeTrans': 'Translate the following Java code into C#: ',
            'CodeSearchNet': 'Summarize the following Ruby code into English: ',
            'BFP': 'Refactor or improve the following Java code: '}

HUGGINGFACE_DATASET = {'CONCODE': 'AhmedSSoliman/CodeXGLUE-CONCODE',
                    'CodeTrans': 'CM/codexglue_codetrans',
                    'CodeSearchNet': 'semeru/code-text-ruby',
                    'BFP': 'ayeshgk/code_x_glue_cc_code_refinement_annotated'}


def convert_to_codetask(split_name="train"):
    for task in TASK_LIST:
        save_dir = os.path.join(FOLDER_NAME, task)
        os.makedirs(save_dir, exist_ok=True)
        dataset = load_dataset(HUGGINGFACE_DATASET[task], split=split_name)
        
        output_data = {
            "Definition": [DEFINITION[task]],
            "Positive Examples": [],  # Optional
            "Negative Examples": [],  # Optional
            "Instances": []
        }

        for i, example in enumerate(tqdm(dataset, desc=f"Processing {split_name}")):
            input_text = example[TEXT_KEYS[task]]
            output_text = example[LABEL_KEYS[task]]
            
            uid = hashlib.md5(input_text.encode("utf-8")).hexdigest()

            output_data["Instances"].append({
                "id": f"{task}-{uid}",
                "input": input_text,
                "output": [output_text]
            })

        with open(os.path.join(save_dir, f"{split_name}.json"), "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

# 🔁 Run for all splits
for split in ["train", "validation", "test"]:
    try:
        convert_to_codetask(split)
    except Exception as e:
        print(f"⚠️  Skipped {split}: {e}")
