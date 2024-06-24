import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from opt import get_args

modes = ["zero-shot","few-shot","AT-zero-shot","AT-few-shot","CoT-zero-shot","CoT-few-shot","AT-CoT-zero-shot","AT-CoT-few-shot"]

def filter_based_on_the_last_marker(s,marker="[CQ]"):
    for i in range(len(s)-len(marker),-1,-1):
        if s[i:i+len(marker)] == marker:
            return s[i:]
    raise IndexError(f"no marker found: {marker}.")

def clean_sentence(s,step):
    if step == 1:
        if "[CQ]" in s:
            s = filter_based_on_the_last_marker(s)
        tag = True
        for i in range(1,6):
            tag = tag and f"({i})" in s
        if tag:
            ixs, cqs = [], []
            for i in range(1,6):
                ixs.append(s.index(f"({i})"))
            ixs.append(len(s))
            for i in range(5):
                cqs.append(s[ixs[i]+3:ixs[i+1]].replace("[CQ]","").split('\n')[0].strip())
        else:
            tag = True
            for i in range(1,6):
                tag = tag and f"[{i}]" in s
            ixs, cqs = [], []
            if tag:
                for i in range(1,6):
                    ixs.append(s.index(f"[{i}]"))
                ixs.append(len(s))
                for i in range(5):
                    cqs.append(s[ixs[i]+3:ixs[i+1]].replace("[CQ]","").split('\n')[0].strip())
        return [q.replace("[CQ]",'').replace("[/CQ]",'').strip().replace('"','') for q in cqs]
    else:
        if "[Q]" in s and "[/Q]" in s:
            s = filter_based_on_the_last_marker(s,"[Q]")
            i1, i2 = s.index("[Q]"), s.index("[/Q]")
            return s[i1+3:i2].strip('\n').strip()
        elif "[Q]" in s:
            i1 = s.index("[Q]")
            return s[i1+3:].split('\n')[0].split()
        else:
            return ""

def clean(docs,step):
    res = []
    for doc in docs:
        s = doc[0]["generated_text"].split("<|end_header_id|>")[-1].strip("\n").strip()
        res.append(clean_sentence(s,step))
    return res

def save_as_csv_files(args,qs,cqs,modes,intentions=[]):
    if args.step == 1:
        for mode in modes:
            assert len(qs) == len(cqs[mode]), f"ERROR {mode}: number of originals queries should be equal to the number of lists of clarification queries."
        for mode in modes:
            copied_qs, copied_cqs = [], []
            for q, tmp_cqs in zip(qs,cqs[mode]):
                copied_qs += [q] * len(tmp_cqs)
                copied_cqs += tmp_cqs
            df_res = pd.DataFrame({"q":copied_qs,"cq1":copied_cqs})
            df_res.to_csv(os.path.join(args.output_dir,f"df_cqs_p{args.step}_{args.dataset_name}_{mode}.csv"),index=False)
        print(f"process finished: results saved as .csv files.")
    else:
        for mode in modes:
            assert len(qs) == len(intentions), f"ERROR {mode}: number of originals queries should be equal to the number of lists of intentions."
            assert sum(list(map(len,intentions))) == len(cqs[mode]), f"ERROR {mode}: number of selected clarifications should be equal to the number of intentions."
        for mode in modes:
            copied_qs, copied_intentions = [], []
            for q, tmp_intentions in zip(qs,intentions):
                copied_qs += [q] * len(tmp_intentions)
                copied_intentions += tmp_intentions
            df_res = pd.DataFrame({'q':copied_qs,"intention":copied_intentions,"bcq1":cqs[mode]})
            df_res.to_csv(os.path.join(args.output_dir,f"df_bcqs_p{args.step}_{args.dataset_name}_{mode}.csv"),index=False)
        print(f"process finished: results saved as .csv files.")

def process_p1(args,modes):
    p1_res = {}
    for mode in modes:
        p1_res[mode] = pickle.load(open(os.path.join(args.output_dir,f"p{args.step}_{args.dataset_name}_{mode}.pkl"),"rb"))
    cqs = {}
    for mode in modes:
        tmp_cqs = clean(p1_res[mode],args.step)
        cq_lengths = list(map(len,tmp_cqs))
        if cq_lengths.count(5) != len(cq_lengths):
            print(mode,"incomplete CQ generation: \n"+'\n'.join([f"# {i} CQ examples: {cq_lengths.count(i)}" for i in range(5) if cq_lengths.count(i)!=0]))
        cqs[mode] = tmp_cqs
    dst_path = os.path.join(args.output_dir,f"p{args.step}_{args.dataset_name}_cqs.json")
    json.dump(cqs,open(dst_path,'w'))
    print(f"process finished: results saved to {dst_path}")
    if args.save_as_csv:
        original_qs = pd.read_csv(os.path.join(args.data_dir,f"df_{args.dataset_name}.csv")).q.values
        if args.dry_run:
            original_qs = original_qs[:5]
        save_as_csv_files(args,original_qs,cqs,modes)
    return

def process_p2(args,modes):
    p2_res = {}
    for mode in modes:
        p2_res[mode] = pickle.load(open(os.path.join(args.output_dir,f"p{args.step}_{args.dataset_name}_{mode}.pkl"),"rb"))
    bcqs = {}
    for mode in modes:
        bcqs[mode] = clean(p2_res[mode],args.step)
    dst_path = os.path.join(args.output_dir,f"p{args.step}_{args.dataset_name}_bcqs.json")
    json.dump(bcqs,open(dst_path,'w'))
    print(f"process finished: results saved to {dst_path}")
    if args.save_as_csv:
        original_intention = json.load(open(os.path.join(args.data_dir,f"{args.dataset_name}_intention.json")))
        original_qs = original_intention['q']
        intentions = original_intention["intention"]
        if args.dry_run:
            original_qs, intentions = original_qs[:5], intentions[:5]
        save_as_csv_files(args,original_qs,bcqs,modes,intentions)
    return

def process(args,modes):
    if args.step == 1:
        process_p1(args,modes)
    elif args.step == 2:
        process_p2(args,modes)
    else:
        raise ValueError("step can only be 1 or 2.")

if __name__ == "__main__":
    args = get_args()
    if args.prompt_type:
        process(args,[args.prompt_type])
    else:
        process(args,modes)

