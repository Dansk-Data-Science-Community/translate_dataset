from dataclasses import dataclass
from typing import List, Literal, Optional, Callable
import datasets
from vllm import LLM, SamplingParams


@dataclass
class ColumnSpec:
    name: str
    kind: Literal["string", "list", "dict_list"]
    key: Optional[str] = None

PromptBuilder = Callable[[str, str, str], str]

def default_prompt_builder(text: str, source_language: str, target_language: str) -> str:
    return (
        f"Translate this text from {source_language} to {target_language}. "
        f"Return only the translated text. Nothing else.\n*TEXT:*\n{text}"
    )

class Translator():

    def __init__(
        self,
        use_api: bool,
        cols_to_translate: List[ColumnSpec],
        hf_model_id: str,
        source_language: str,
        target_language: str,
        sampling_params: Optional[SamplingParams] = None,
        text_wrapper: PromptBuilder = default_prompt_builder,
    ):
        
        
        self.use_api = use_api
        self.cols_to_translate = cols_to_translate
        self.sampling_params = sampling_params
        self.source_language = source_language
        self.target_language = target_language
        self.text_wrapper = text_wrapper

        # Only instantiate LLM if we're not in API mode
        self.llm = None
        if not use_api and hf_model_id is not None:
            self.llm = LLM(model=hf_model_id)


    @classmethod
    def from_model_id(
        cls,
        hf_model_id: str,
        cols_to_translate: List[ColumnSpec],
        source_language: str,
        target_language: str,
        sampling_params: SamplingParams,
        use_api: bool = False,
        text_wrapper=default_prompt_builder,
        **llm_kwargs,
    ):
        """
        Example usage:
        
        ```python
        translator = Translator.from_model_id(
        hf_model_id="google/gemma-3-27b-it",
        cols_to_translate=cols_to_translate,
        source_language="English",
        target_language="Danish",
        sampling_params=sampling_params,
        max_seq_len_to_capture=8000,
        )
        ```
        """
        llm = LLM(model=hf_model_id, **llm_kwargs)

        return cls(
            use_api=use_api,
            cols_to_translate=cols_to_translate,
            source_language=source_language,
            target_language=target_language,
            sampling_params=sampling_params,
            llm=llm,
            text_wrapper=text_wrapper,
        )

    def call_translation_model_api(self, texts: list[str]) -> list[str]:
        """
        Used to call a translation API like DeepL.
        """
        raise NotImplementedError("Implement API client here")
        
    def call_llm_non_api(self, texts: list[str]) -> list[str]:
        """
        Call LLM object defined in this script.
        """
        if self.llm is None:
            raise RuntimeError("LLM backend not initialized")

        prompts = [
            self.text_wrapper(text, self.source_language, self.target_language)
            for text in texts
        ]

        llm_output = self.llm.chat(prompts, self.sampling_params)
        return [output.outputs[0].text for output in llm_output]

    def _translate_texts(self, texts: list[str]) -> list[str]:
        if self.use_api:
            return self.call_translation_model_api(texts)
        else:
            return self.call_llm_non_api(texts)
    
    def _translate_dict_column(self, values: list[dict], key_to_translate: str) -> list[dict]:
        """
        This function is designed to translate a column in a batch where each element in the column is a dict,
        and what you want to translate is a list in the dict

        E.g. in MS Marco:
        Loop the rows in the "passages" column. Each element in the column is a dict where "passage_text" is a list of strings that we want to translate
        """
        result = []
        
        # values: list[dict], each dict[key_to_translate] is list[str]
        for row in values:
            texts = row[key_to_translate]
            translated = self._translate_texts(texts)
            row_copy = {**row, key_to_translate: translated}
            result.append(row_copy)
        return result

    def _translate_string_column(self, values: list[str]) -> list[str]:
        """
        Translate a column in a batch where each row in the column is a string element.
        Used for the "query" column in MS Marco
        """
        # values: list[str]
        return self._translate_texts(values)
    
    def _translate_list_column(self, values: list[list[str]]) -> list[list[str]]:
        """
        Translate a column in a batch where each row in the column is a list element. 
        Used for the "answer" column in MS Marco where each element is a list
        """
        # values: list[list[str]]
        return [self._translate_texts(lst) for lst in values]
    
    def translate_batch(self, batch):

        for spec in self.cols_to_translate:
            values = batch[spec.name]

            if spec.kind == "string":
                translated = self._translate_string_column(values)
            elif spec.kind == "list":
                translated = self._translate_list_column(values)
            elif spec.kind == "dict_list":
                translated = self._translate_dict_column(values, spec.key)

            batch[spec.name + "_translated"] = translated

        return batch

    def translate(self, dataset):
        return dataset.map(
            self.translate_batch,
            batched=True,
            batch_size=2,
            num_proc=None,
        )


class DeeplTranslator(Translator):
    def __init__(self, deepl_client, **kwargs):
        super().__init__(use_api=True, hf_model_id=None, **kwargs)
        self.deepl_client = deepl_client

    def call_translation_model_api(self, texts: list[str]) -> list[str]:
        return self.deepl_client.translate_texts(
            texts=texts,
            source_lang=self.source_language,
            target_lang=self.target_language,
        )


