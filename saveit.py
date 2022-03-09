# This is used to make the aitextgen gpt2 model compatable with huggingface
# and the perplexity.py code

from aitextgen import aitextgen


ai = aitextgen(model_folder="trained_model",
                tokenizer_file="aitextgen.tokenizer.json")

ai.save_for_upload("trained_model")
