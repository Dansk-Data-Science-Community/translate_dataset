from translators import Translator, ColumnSpec
import datasets
from vllm import SamplingParams

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

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8000,
    top_p=1.0,
)

translator = Translator(
    use_api=False,
    cols_to_translate=cols_to_translate,
    hf_model_id="google/gemma-3-12b-it",
    source_language="English",
    target_language="Danish",
    sampling_params=sampling_params
)

translated_ds = translator.translate(ds)

translated_ds.to_csv("ms-marco-danish.csv")

translated_ds.push_to_hub("DDSC/ms-marco-danish")

