import transformers
import torch
from langchain.llms import HuggingFacePipeline


class LangchainMPTWrapper(HuggingFacePipeline):
    def __init__(
            self,
            model_path,
            max_seq_len=1024,
            max_new_tokens=126,
            temperature=0.1,
            top_p=0.15,
            top_k=0,
            repetition_penalty=1.1,
            device="cpu",
            torch_dtype=torch.bfloat16,
            attn_impl='triton'
    ):
        tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

        stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

        class StopOnTokens(transformers.StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                for stop_id in stop_token_ids:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False

        config = transformers.AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        config.attn_config['attn_impl'] = attn_impl
        config.update({"max_seq_len": int(max_seq_len)})

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        model.to(device)
        print(f"Model loaded on {device}")
        stopping_criteria = transformers.StoppingCriteriaList([StopOnTokens()])
        generate_text = transformers.pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=True,
            task='text-generation',
            device=device,
            stopping_criteria=stopping_criteria,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=float(top_k),
            max_new_tokens=int(max_new_tokens),
            repetition_penalty=float(repetition_penalty)
        )
        super().__init__(pipeline=generate_text)