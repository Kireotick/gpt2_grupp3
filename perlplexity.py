from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch


device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model_id is used when calling on a pretrained model from huggingface
model_id = "gpt2-large"
PATH = 'trained_model'

# This following code can either call on a pretrained model or tokenizer,
# which is currenty commented. Otherwise it uses the model or toeknizer in
# the specified PATH, which is the one we trained on the pod_dialogue.txt

model = GPT2LMHeadModel.from_pretrained(PATH, local_files_only=True).to(device)
#model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(PATH, local_files_only=True)
#tokenizer = GPT2TokenizerFast.from_pretrained(model_id)


# Code bellow determines which txt file should be used to evaluate perplexity

#test = open("pod_dialogues.txt", encoding = 'utf-8')
test = open("movie_dialogues.txt", encoding = 'utf-8')

encodings = tokenizer("\n\n".join(test), return_tensors="pt")

from tqdm import tqdm

max_length = model.config.n_positions
stride = 512
nlls = []
for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, encodings.input_ids.size(1))
    trg_len = end_loc - i  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs[0] * trg_len

    nlls.append(neg_log_likelihood)
ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
print(ppl)
