from datasets import load_dataset
import datasets
from typing import List
from vllm import LLM, SamplingParams
from copy import copy 
from dataclasses import dataclass

@dataclass
class TranslationPrompt:

    source_lang: str
    target_lang: str
    input_text: str

    def render(self):
        return (
            f"Translate this text from {self.source_lang} "
            f"to {self.target_lang}. You return only the translated text."
            f"Nothing else.\n*TEXT:*\n {self.input_text}"
        )

class Translator():

    def __init__(self, 
                 use_api: bool, 
                 cols_to_translate: List, 
                 hf_model_id: str, 
                 prompt_template: str, 
                 source_language, 
                 target_language,
                 dataset: datasets.Dataset):
        
        self.use_api = use_api
        self.cols_to_translate = cols_to_translate
        self.llm = LLM(model=hf_model_id, max_seq_len_to_capture=8000)
        self.prompt_template = prompt_template
        self.source_langauge = source_language
        self.target_language = target_language,
        self.dataset = dataset
        
    def call_llm_non_api(self, texts):
        """
        Call LLM object defined in this script, i.e. not an API endpoint that exposes an LLM
        """

        prompts = [
            TranslationPrompt(
                input_text=text,
                target_lang=self.target_language,
                source_lang=self.source_language
                )
            for text in texts
            ]
        
        llm_output = self.llm.chat(prompts, self.sampling_params)
        
        return llm_output
    
    def get_str_from_llm_output(self, llm_output):

        return [output.outputs[0].text for output in llm_output]

    def translate_passages_column(self, batch: datasets.formatting.formatting.LazyBatch) -> datasets.formatting.formatting.LazyBatch:
        """
        Loop the rows in the "passages" column. Each element in the column is a dict where "passage_text" is a list of strings that we want to translate
        """
        result = []

        # Loop over the "passages" column which is a list. Every element in the list is a dictionary
        for ls in batch['passages']: 

            # Access the relevant key of the dictionary to get a list
            list_of_texts = ls["passage_text"]

            # Translate the list
            llm_output = self.call_llm_non_api(list_of_texts)

            # Post process the raw LLM output to get a list of strings instead of a list of LLM outputs
            llm_output = self.get_str_from_llm_output(llm_output)

            # Replace the list in the dictionary with the translated texts
            ls_copy = copy(ls)

            ls_copy["passage_text"] = llm_output

            # Append list to list
            result.append(ls_copy)

        return result
        
    def translate_query_column(self, batch: datasets.formatting.formatting.LazyBatch) -> datasets.formatting.formatting.LazyBatch:

        llm_output = self.call_llm_non_api(texts=batch["query"])
        llm_output = self.get_str_from_llm_output(llm_output)
    
        return llm_output
    
    def translate_answers_column(self, batch: datasets.formatting.formatting.LazyBatch) -> list:

        result = []

        for ls in batch["answers"]:

            llm_output = self.call_llm_non_api(texts=ls)

            llm_output = self.get_str_from_llm_output(llm_output)

            result.append(llm_output)
        
        return result
    
    def map_cols_to_get_text_method(self, col_type):

        return {
            list : self.get_string_from_list, # used for "answer" col in MS Marco
            str : self.get_string_from_str, # used for "query" col in MS Marco
            dict : self.get_string_from_dict, # used for "passages" col in MS Marco
        }

    def map_cols_to_translation_method(self, col_type):
        """
        Translation methods must return an object that is of the same type and structure as the original column
        E.g. if batch["columnA"] is a list of dictionaries, the translation method for "columnA" must also return a list of dictionaries

        """
        dic = {
            list : self.translate_answers_column, # used for "answer" col in MS Marco
            str : self.translate_query_column, # used for "query" col in MS Marco
            dict : self.translate_passages_column, # used for "passages" col in MS Marco
        }

        return dic[col_type]
    
    def translate_batch(self, batch):

        # batch is a dict: column_name -> list of values
        for col in self.cols_to_translate:

            col_type = type(batch[0][col]) # get col type. Assume we can access the column type by indexing batch
            translation_method = self.map_cols_to_translation_method(col_type)
            
            # Receive a list of objects that look like the original column
            translations = translation_method(batch[col]) # passages will receive dict, answers will receive list of strings, query will receive string

            batch[col + "_translated"] = translations

        return batch

    def run(self, dataset):

        translated_dataset = dataset.map(
            self.translate_batch,
            batched=True,
            batch_size=32,   # tune based on GPU/memory/API limits
            num_proc=4,      # optional parallelism
        )

        return translated_dataset

