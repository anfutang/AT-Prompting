from argparse import ArgumentParser

def get_args():
     parser = ArgumentParser(description="prompting with ambiguity type definitions.")
	
     parser.add_argument("--data_dir",type=str,default="./data/")
     parser.add_argument("--output_dir",type=str,default="./output/")
     parser.add_argument("--dataset_name",type=str)
     parser.add_argument("--model_name",type=str,default="meta-llama/Llama-2-13b-chat-hf")
     parser.add_argument("--prompt_type",type=str,default="")
     parser.add_argument("--seed",type=int,default=50)
     parser.add_argument("--batch_size",type=int,default=2,help="batch size used for inference")
     parser.add_argument("--step",type=int,default=1,help="number of processing stage: 1 or 2.")
     parser.add_argument("--save_as_csv",action="store_true")
     parser.add_argument("--dry_run",type=bool,default=False,help="set dry_run for a quick test (over 5 examples).")
     args = parser.parse_args()
     return args