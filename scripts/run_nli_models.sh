#!/usr/bin/env bash
model_name="doc-nli"  # choose from doc-nli, mnli, nq-nli
model_dir="./nli_models"
qa_nli_model_path="${model_dir}/nq-nli.tar.gz"
mnli_model_path="${model_dir}/mnli.tar.gz"
doc_nli_model_path="${model_dir}/doc-nli.tar.gz"

if [[ ! -d ${model_dir} ]]; then
    mkdir ${model_dir}
else
    echo "${model_dir} already exist"
fi

if [[ ! -f ${qa_nli_model_path} ]]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fhP7BEV47XAC6-vHF8euOLmRW6YH3a50' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fhP7BEV47XAC6-vHF8euOLmRW6YH3a50" -O "${qa_nli_model_path}" && rm -rf /tmp/cookies.txt
fi

if [[ ! -f ${mnli_model_path} ]]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1t-qTGKw46VVfTn51fyf25LMDkBdCmM3d' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1t-qTGKw46VVfTn51fyf25LMDkBdCmM3d" -O "${mnli_model_path}" && rm -rf /tmp/cookies.txt
fi

if [[ ! -f ${doc_nli_model_path} ]]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1c5fgAPrrA-6mb8ZqLgkv0R2IokoRxmfE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1c5fgAPrrA-6mb8ZqLgkv0R2IokoRxmfE" -O "${doc_nli_model_path}" && rm -rf /tmp/cookies.txt
fi

python3 -m scripts.run_nli  --model ${model_name}