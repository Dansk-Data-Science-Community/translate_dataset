
cols_to_translate = ["answers", "passages", "query"]

dataset = load_dataset(
    "microsoft/ms_marco",
    name="v1.1",
    split="train"
)

print(dataset)


dataset[1]["answers"][0]
type(dataset[1]["passages"])

type(dataset[1]["answers"])




def generate_non_api():

    outputs = LLM.chat(prompts, sampling_params)

    return None



def translate_ms_marco(row):

    # To access the string: dataset[row_idx]["answers"][0]
    answer = row[0]

    answer = translate(answer)


def test():
    print("hej")

dic = {list : test}

el = []

el_type = type(el)

func = dic[el_type]

func()

el_type == list

batch["passages"].keys()

passage_text = batch["passages"]["passage_text"]

len(passage_text)

batch_translated = batch
batch_translated["passages"]["passage_text"] = ["text", "text"]


template = "Hej {input_text}"
fill = "Joe"
template_filled = template.replace("{input_text}", fill)

template

type(dataset)

def func(batch):

    print(type(batch))
    print(f"Type of batch['passages'][0]): {type(batch['passages'][0])}")
    print(f"Type of batch['passages']: {type(batch['passages'])}")
    print("------------------")

dataset.map(
            func,
            batched=True,
            batch_size=32,   # tune based on GPU/memory/API limits
            num_proc=4,      # optional parallelism
        )


type(dataset)