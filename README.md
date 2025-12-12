# translate_dataset
A project for translating datasets to Danish

# To do
- The code runs. Quality check the output
- Select best translation model: Translate the [Skolegpt-instruct](https://huggingface.co/datasets/kobprof/skolegpt-instruct) subset of [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) from English to Danish with several candidate models (e.g. Llama 70b, Gemma 3 27b, Qwen3 32b etc) and select the model that produces the best translations (eg the one with the best Comet scores or the shortest Libre distance) where the Skolegpt-instruct is the reference translations.
- Translate MS Marco using the `run.py` script and the selected translation model
    
# Setup
To set up a server with the pre-requisite software, run the `setup_server.sh` script like so:
```bash
source setup_server.sh
```
Note that you will be prompted to enter a Hugging Face access token in order to access any gated repo that your code uses.  