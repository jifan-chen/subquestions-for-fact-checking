#!/usr/bin/env bash
working_dir="/mnt/data1/jfchen/fine_grained_fact_verification/"
#working_dir="./qg_data"
data_url="https://docs.google.com/spreadsheets/d/13D_X4CNP83s7EGHieqBXLX7ypLLb-upTdcmYReTAzws/export?format=csv&gid=1646637005"
ds_config_path="./src/seq2seq_converter/ds_config.json"
dataset_path="${working_dir}/train.csv"
output_path="${dataset_path}_predictions.csv"
output_dir="${working_dir}/seq2seq2_model_output/qg-nucleus-ruling-t5-3b"
# if training, set model to this:
model="t5-3b"

if [[ ! -d ${working_dir} ]]; then
    mkdir ${working_dir}
else
    echo "${working_dir} already exist"
fi

echo "Downloading data ........"
wget -O ${dataset_path} ${data_url}

echo "Running $1"
python3 -m src.seq2seq_converter.seq2seq_converter \
	--model_name_or_path ${model} \
	--do_train True \
	--do_eval False \
	--do_predict False \
	--output_dir ${output_dir} \
	--task $1 \
	--per_device_train_batch_size=8 \
	--per_device_eval_batch_size=8 \
	--overwrite_output_dir True\
	--predict_with_generate \
	--overwrite_cache True \
	--max_source_length 256 \
	--max_target_length 128 \
	--pad_to_max_length False \
	--output_path ${output_path} \
	--output_format csv \
	--train_file ${dataset_path} \
	--prediction_file ${dataset_path} \
	--validation_file ${dataset_path} \
	--data_source dummy \
	--num_train_epochs 8 \
	--save_steps 2000 \
	--deepspeed ${ds_config_path} \
	--fp16 True \

