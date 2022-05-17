#!/usr/bin/env bash
data_dir="./ClaimDecomp"
train_path=${data_dir}/train.jsonl
dev_path=${data_dir}/dev.jsonl
test_path=${data_dir}/test.jsonl
train_url='https://www.cs.utexas.edu/~jfchen/claim-decomp/train.jsonl'
dev_url='https://www.cs.utexas.edu/~jfchen/claim-decomp/dev.jsonl'
test_url='https://www.cs.utexas.edu/~jfchen/claim-decomp/test.jsonl'
if [[ ! -d ${data_dir} ]]; then
    mkdir ${data_dir}
else
    echo "${data_dir} already exist"
fi

wget -O  ${train_path} ${train_url}
wget -O  ${dev_path}   ${dev_url}
wget -O  ${test_path}  ${test_url}

echo "reconstructing the dataset ..."
python3 scripts/reconstruct_dataset.py  --input_path ${train_path} --output ${train_path}
python3 scripts/reconstruct_dataset.py  --input_path ${dev_path} --output ${dev_path}
python3 scripts/reconstruct_dataset.py  --input_path ${test_path} --output ${test_path}