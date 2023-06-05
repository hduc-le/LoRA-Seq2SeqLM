from .consts import DEFAULT_INPUT_MODEL, NEW_LINE
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_tokenizer(model_name_or_path: str = DEFAULT_INPUT_MODEL, additional_special_tokens: list = [NEW_LINE]):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
    return tokenizer, added_tokens

def load_model(model_name_or_path: str = DEFAULT_INPUT_MODEL):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    return model

def load_model_tokenizer(model_name_or_path: str = DEFAULT_INPUT_MODEL, additional_special_tokens: list = [NEW_LINE]):
    tokenizer, added_tokens = load_tokenizer(model_name_or_path, additional_special_tokens)
    model = load_model(model_name_or_path)

    if added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def generate(prompt, model, tokenizer, **generate_kwargs):
    prompt_encodings = tokenizer(prompt, return_tensors="pt")
    input_ids = prompt_encodings.input_ids.to(model.device)
    attention_mask = prompt_encodings.attention_mask.to(model.device)

    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded #decoded[len(prompt) :]

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model (nn.Module).
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )