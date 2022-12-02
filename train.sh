# 1、为预训练准备数据，结果在temp/tmp_data下
echo "准备预训练所需要数据"
python code/process_data.py


#2、 预训练开始  取100条测试，记得修改回来
# 模型结果存放在/home/mw/project/data/pretrain_model_mlm下
echo "开始预训练"
python code/run_pretrain_mlm.py


# 3、生成训练数据集
python code/data_process.py train


# 4、训练全量单模
'''
训练8轮，取后六轮平均模型参数
存储到/home/mw/temp/model_data/gp_output/model_all_1
'''
python code/train.py



# 5、训练十折模型
'''
十折
每折训练8轮，取后六轮平均模型参数
存储到/home/mw/temp/model_data/gp_output/model_10fold
'''
python code/train.py --output_dir /home/mw/temp/model_data/gp_output/model_10fold --kfold True

# 6、制作30w伪标
'''
利用十折和全量模型制作30w伪标
存储到/home/mw/temp/model_data/gp_output/model_10fold
'''
python code/stacking.py


# 7、30伪标+全量
'''
伪标参与训练
存储到/home/mw/temp/model_data/gp_output/best_model
'''
python code/train.py --output_dir /home/mw/temp/model_data/gp_output/best_model --epoch 5 --pseudos True