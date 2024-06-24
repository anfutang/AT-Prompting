import os
import pandas as pd
import json
import pickle
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from opt import get_args

if __name__ == "__main__":
    args = get_args()

    model_id = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_id,padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size
    )

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    prompt = json.load(open(os.path.join(args.data_dir,"prompt-2.json")))["prompt"]
    data = json.load(open(os.path.join(args.data_dir,f"{args.dataset_name}_intention.json")))
    cqs = json.load(open(os.path.join(args.output_dir,f"p1_{args.dataset_name}_cqs.json")))[args.prompt_type]
    qs = data['q']
    intentions = data["intention"]

    if args.dry_run:
        qs, intentions, cqs = qs[:5], intentions[:5], cqs[:5]
    
    ps = []
    for q, tmp_intentions, tmp_cqs in zip(qs,intentions,cqs):
        enumerated_cqs = '\n'.join([f"({i}) {tmp_cqs[i]}" for i in range(len(tmp_cqs))])
        for intention in tmp_intentions:
            p = prompt.replace("{{QUERY}}",q).replace("{{INTENTION}}",intention).replace("{{CQ}}",enumerated_cqs)
            ps.append(p)
    
    outputs = pipeline(
        ps,
        max_new_tokens=2000,
        eos_token_id=terminators,
        do_sample=False,
    )

    with open(os.path.join(args.output_dir,f"p2_{args.dataset_name}_{args.prompt_type}.pkl"),"wb") as f:
        pickle.dump(outputs,f,pickle.HIGHEST_PROTOCOL)

    
