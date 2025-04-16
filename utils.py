
from transformers import AutoTokenizer, PreTrainedTokenizer
from trl import ModelConfig

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

PAD_TOKEN_MAP = {
    "DeepSeek-R1-Distill-Llama-8B": "<|finetune_right_pad_id|>",
    "DeepSeek-R1-Distill-Qwen-1.5B": "<|video_pad|>",
    "Qwen2.5-3B": "<|video_pad|>",
    "Qwen2.5-7B": "<|video_pad|>",
    "Qwen2.5-14B": "<|video_pad|>",
    "Qwen2.5-32B": "<|video_pad|>",
}

def get_tokenizer(
    model_args: ModelConfig, chat_template: str | None = None, auto_set_chat_template: bool = True
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if chat_template is not None:
        tokenizer.chat_template = chat_template
    elif auto_set_chat_template and tokenizer.get_chat_template() is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    model_name = model_args.model_name_or_path.split("/")[-1]    
    if model_name in PAD_TOKEN_MAP:
        pad_token = PAD_TOKEN_MAP[model_name]
        tokenizer.pad_token = pad_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)

    assert tokenizer.pad_token is not None 
    assert tokenizer.pad_token != tokenizer.eos_token and tokenizer.pad_token_id != tokenizer.eos_token_id

    return tokenizer

def test_vllm_serve(model_name_or_path):
    from openai import OpenAI

    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    completion = client.completions.create(model=model_name_or_path, prompt="San Francisco is a")
    print("Completion result:", completion)

if __name__ == "__main__":
    test_vllm_serve("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")