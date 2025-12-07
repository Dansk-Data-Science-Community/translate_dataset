from translators import Translator, ColumnSpec
import datasets


# How to use   
cols_to_translate = [
    ColumnSpec(name="query", kind="string"),
    ColumnSpec(name="answers", kind="list"),
    ColumnSpec(name="passages", kind="dict_list", key="passage_text"),
]

translator = Translator(
    use_api=False,
    cols_to_translate=cols_to_translate,
    hf_model_id="google/gemma-3-27b-it",
    source_language="English",
    target_language="Danish",
)

ds = datasets.load_dataset("microsoft/ms_marco", split="train")
translated_ds = translator.translate(ds)
