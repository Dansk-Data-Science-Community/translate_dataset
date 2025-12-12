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
    hf_model_id="google/gemma-3-12b-it",
    source_language="English",
    target_language="Danish",
)

#ds = datasets.load_dataset("microsoft/ms_marco", 'v1.1', split="train")
#load_dataset('microsoft/ms_marco', 'v1.1')`


translated_ds = translator.translate(ds)

translated_ds.push_to_hub("ThatsGroes/ms-marco-danish")

translated_ds.to_csv("ms-marco-danish.csv")

ls = [
"The RBA's outstanding reputation has been affected by the 'Securency",
"The Reserve Bank of Australia (RBA) came into being on 14",
"RBA Recognized with the 2014 Microsoft US Regional Partner of the",
"The inner workings of a rebuildable atomizer are surprisingly simple. The coil inside the",
"Results-Based Accountability® (also known as RBA) is a disciplined way",
"Results-Based Accountability® (also known as RBA) is a disciplined way",
"RBA uses a data-driven, decision-making process to help communities and",
"NetIQ Identity Manager. Risk-based authentication (RBA) is a method",
"A rebuildable atomizer (RBA), often referred to as simply a “re",
"Get To Know Us. RBA is a digital and technology consultancy with roots in"
]

for i in ls:
    print(len(i))