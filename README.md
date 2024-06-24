# AT-Prompting

- launch a prompt (e.g. few-shot on trec_web_track; step-1 prompting on IR tasks)
```
sbatch exp-fs.sh trec_web_track 1 
```

- parse the output
```
python3 process_output.py --dataset_name trec_web_track --prompt_type few-shot --save_as_csv --step 1
```
