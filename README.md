# translate_dataset
A project for translating datasets to Danish

# To do
- make generic base "translator" class in "playground.py" and rename the file to something more sensible. The base class shuld be designed to take translate one or more string columns in a HF dataset. Make it easy to extend to specialized cases such as MS marco where the columns we want to translate or not only string columns, but also columns with lists or json  
- make MS marco specific child class of "translator"
    

# notes 
flow:
- use map on dataset to apply function. 
- inside function, loop over columns. apply designated function to each column to extract string elements from row 
- store translation in "_translated" 

from datasets import load_dataset

dataset = load_dataset("your/dataset")

text_cols = ["title", "abstract", "body"]

def translate_many(texts):
    # texts is a list of strings
    # Call your translation model/API here in batch.
    # Example (pseudo-code):
    # results = translation_pipeline(texts)  # list of dicts or strings
    # return [r["translation_text"] for r in results]
    ...
    

def translate_batch(batch):
    # batch is a dict: column_name -> list of values
    for col in text_cols:
        translated = translate_many(batch[col])
        batch[col + "_translated"] = translated
    return batch

translated_dataset = dataset.map(
    translate_batch,
    batched=True,
    batch_size=32,   # tune based on GPU/memory/API limits
    num_proc=4,      # optional parallelism
)

