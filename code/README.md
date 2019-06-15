### 环境要求

- Keras with TensorFlow backend
- [Keras-VggFace](https://github.com/rcmalli/keras-vggface) - `pip install keras_vggface`
- `python 2.7`

### 训练hydxj_train.py

命令行输入

```
python hydxj_train.py --training_dataset='./af2019-ksyun-training-20190416' \
  --nb_class=85 \
  --model='./model/hydxj_model3.h5' \
  --epoch=4
```

1. `training_dataset`为`annotations.csv`所在目录，训练集和校验集会在此目录生成，分别为`training.txt`和`validation.txt`
2. `nb_class`为训练人物数量，默认为85
3. `model`为模型存储路径
4. `epoch`为训练集训练轮次，默认为2

**我们的python版本为2.7，csv生成predictions.csv在转换为utf-8编码遇到版本问题，时间紧迫，利用文本编辑工具转码为utf-8**

### 测试hydxj_test.py

命令行输入

```
python hydxj_test.py --test_dataset='./af2019-ksyun-testB-20190424' \
  --model='./model/hydxj_model2.h5' \
  --prediction_file='./result/predictions.csv'
```

1. 测试集目录(`$TESTSET_DIR`)，即`list.csv`文件所在目录，通过参数`--test_dataset=$TESTSET_DIR`传入程序，如果输入`$TESTSET_DIR`中包含了`.txt`则输入了测试集文件，这时需要设置通过`--test_dir`图片所在目录否则会出错。反之为目录路径，先根据`list.csv`在该目录下生成`test.txt`文件，再根据该测试集文件测试
2. 预测所用模型所在目录路径(`$MODEL_DIR`)，通过参数 `--model=$MODEL_DIR`传入程序
3. 预测结果写入csv文件，文件路径通过参数`--prediction_file=$PREDICTION_FILE`指定