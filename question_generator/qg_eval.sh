#!/usr/bin/env bash
# positional arguments: 1->task, 2->model_dir, 3->output_path
working_dir="./qg_data"
ds_config_path="./src/seq2seq_converter/ds_config.json"
data_url="https://docs.google.com/spreadsheets/d/13D_X4CNP83s7EGHieqBXLX7ypLLb-upTdcmYReTAzws/export?format=csv&gid=581157458"
dataset_path="${working_dir}/eval.csv"
output_path=${3}
output_dir="${2}"

# if training, set model to this:
model="${2}"

if [[ ! -d ${working_dir} ]]; then
    mkdir ${working_dir}
else
    echo "${working_dir} already exist"
fi

echo "Downloading data ........"
wget -O ${dataset_path} ${data_url}

echo "Inference on $1"
python3 -m src.seq2seq_converter.seq2seq_converter \
	--model_name_or_path ${model} \
	--do_train False \
	--do_eval False \
	--do_predict True \
	--output_dir ${output_dir} \
	--task $1 \
	--per_device_train_batch_size=8 \
	--per_device_eval_batch_size=8 \
	--overwrite_output_dir \
	--predict_with_generate \
	--overwrite_cache True \
	--max_source_length 256 \
	--max_target_length 512 \
	--pad_to_max_length False \
	--output_path ${output_path} \
	--output_format csv \
	--train_file ${dataset_path} \
	--prediction_file ${dataset_path} \
	--validation_file ${dataset_path} \
	--data_source dummy \
	--num_train_epochs 30 \
	--beam_size 5 \
#	--deepspeed ${ds_config_path} \
#	--fp16 True


