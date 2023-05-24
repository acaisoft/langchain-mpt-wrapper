import transformers
import torch
from langchain.llms import HuggingFacePipeline


class LangchainMPTWrapper(HuggingFacePipeline):
    def __init__(self, model_path, max_seq_len, max_new_tokens, temperature, top_p, top_k, repetition_penalty, device, torch_dtype, attn_impl='triton'):
        tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

        # mtp-7b is trained to add "<|endoftext|>" at the end of generations
        stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

        # define custom stopping criteria object
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
        # model.eval()
        model.to(device)
        print(f"Model loaded on {device}")

        stopping_criteria = transformers.StoppingCriteriaList([StopOnTokens()])
        generate_text = transformers.pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            device=device,
            # we pass model parameters here too
            stopping_criteria=stopping_criteria,  # without this model will ramble
            temperature=float(temperature),  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            top_p=float(top_p),  # select from top tokens whose probability add up to 15%
            top_k=float(top_k),  # select from top 0 tokens (because zero, relies on top_p)
            max_new_tokens=int(max_new_tokens),  # mex number of tokens to generate in the output
            repetition_penalty=float(repetition_penalty)  # without this output begins repeating
        )
        self.pipeline = generate_text

