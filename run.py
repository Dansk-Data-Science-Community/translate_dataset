from translators import Translator, ColumnSpec
import datasets

# set to >0 to use only the first n_samples of the dataset
n_samples = 10

ds = datasets.load_dataset("microsoft/ms_marco", 'v1.1', split="train")

if n_samples > 0:
    ds = ds.select(range(n_samples))

# How to use   
cols_to_translate = [
    ColumnSpec(name="query", kind="string"),
    ColumnSpec(name="answers", kind="list"),
    ColumnSpec(name="passages", kind="dict_list", key="passage_text"),
]

translator = Translator(
    use_api=False,
    cols_to_translate=cols_to_translate,
    hf_model_id="google/gemma-3-270m-it",
    source_language="English",
    target_language="Danish",
)

#ds = datasets.load_dataset("microsoft/ms_marco", 'v1.1', split="train")
#load_dataset('microsoft/ms_marco', 'v1.1')`


translated_ds = translator.translate(ds)

translated_ds.push_to_hub("ThatsGroes/ms-marco-danish")

translated_ds.to_csv("ms-marco-danish.csv")
