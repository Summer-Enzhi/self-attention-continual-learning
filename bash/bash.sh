# SKIN8
nohup python -u main.py --config options/multi_steps/dynamic_er/skin8.yaml --device 3 >./logs_train_single_validTest_sofar_f1/skin8_DF_source/dynamicer.log 2>&1 &

nohup python -u main.py --config options/multi_steps/acl/skin8.yaml --device 0 >./logs_train_single_validTest_sofar_f1/skin8_DF_source/no_load.log 2>&1 &

nohup python -u main.py --config options/multi_steps/acl/skin8.yaml --device 2 >./logs_train_single_validTest_sofar_f1/skin8_DF_source/continual.log 2>&1 &

nohup python -u main.py --config options/multi_steps/acl/skin8.yaml --device 0 >./logs_train_single_validTest_sofar_f1/skin8_DF_source/load_one.log 2>&1 &

nohup python -u main.py --config options/multi_steps/acl/skin8.yaml --device 4 >./logs_train_single_validTest_sofar_f1/skin8_DF_source/attention_2head_drop0.5.log 2>&1 &

nohup python -u main.py --config options/multi_steps/acl/skin8.yaml --device 1 >./logs_train_single_validTest_sofar_f1/skin8_DF_source/onlySelfAttention.log 2>&1 &
nohup python -u main.py --config options/multi_steps/acl/skin8.yaml --device 3 >./logs_train_single_validTest_sofar_f1/skin8_DF_source/source_SelfAttention.log 2>&1 &
nohup python -u main.py --config options/multi_steps/acl/skin8.yaml --device 0 >./logs_train_single_validTest_sofar_f1/skin8_DF_source/source_SelfAttention_RC.log 2>&1 &
nohup python -u main.py --config options/multi_steps/acl/skin8.yaml --device 3 >./logs_train_single_validTest_sofar_f1/skin8_DF_source/source_SelfAttention_LN.log 2>&1 &


# AR
nohup python -u main.py --config options/multi_steps/dynamic_er/ar.yaml --device 3 >./logs_train_single_validTest_sofar_f1/ar/dynamicer.log 2>&1 &

nohup python -u main.py --config options/multi_steps/acl/ar.yaml --device 1 >./logs_train_single_validTest_sofar_f1/ar/no_load.log 2>&1 &

nohup python -u main.py --config options/multi_steps/acl/ar.yaml --device 2 >./logs_train_single_validTest_sofar_f1/ar/continual.log 2>&1 &

nohup python -u main.py --config options/multi_steps/acl/ar.yaml --device 1 >./logs_train_single_validTest_sofar_f1/ar/load_one.log 2>&1 &

nohup python -u main.py --config options/multi_steps/acl/ar.yaml --device 4 >./logs_train_single_validTest_sofar_f1/ar/attention_2head_drop0.5.log 2>&1 &

nohup python -u main.py --config options/multi_steps/acl/ar.yaml --device 5 >./logs_train_single_validTest_sofar_f1/ar/onlySelfAttention.log 2>&1 &
nohup python -u main.py --config options/multi_steps/acl/ar.yaml --device 3 >./logs_train_single_validTest_sofar_f1/ar/source_SelfAttention.log 2>&1 &
nohup python -u main.py --config options/multi_steps/acl/ar.yaml --device 3 >./logs_train_single_validTest_sofar_f1/ar/source_SelfAttention_RC.log 2>&1 &
nohup python -u main.py --config options/multi_steps/acl/ar.yaml --device 3 >./logs_train_single_validTest_sofar_f1/ar/source_SelfAttention_LN.log 2>&1 &


#finetune
nohup python -u main.py --config options/multi_steps/finetune/skin8.yaml --device 0 >./logs_train_single_validTest_sofar_f1/skin8_DF_source/finetune.log 2>&1 &
nohup python -u main.py --config options/multi_steps/finetune/ar.yaml --device 1 >./logs_train_single_validTest_sofar_f1/ar/finetune.log 2>&1 &

# icarl
nohup python -u main.py --config options/multi_steps/icarl/skin8.yaml --device 2 >./logs_train_single_validTest_sofar_f1/skin8_DF_source/icarl.log 2>&1 &
nohup python -u main.py --config options/multi_steps/icarl/ar.yaml --device 3 >./logs_train_single_validTest_sofar_f1/ar/ica
