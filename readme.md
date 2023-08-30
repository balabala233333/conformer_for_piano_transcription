本github当中主要用于复现论文当中的实验数据,本文已将所有的实验结果excel放在此github中:

想要运行您需要一根tesla A40，还有至少1个T的硬盘，并且通过如下指令下载对应的数据集:

```bash
mkdir workspace
cd workspace
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip
wget https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0.zip
wget https://storage.googleapis.com/magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0.zip
unzip *.zip
cd ../
```

然后下载对应的环境，本项目提供了脚本

```bash
conda create --name musicformer
conda activate musicformer
bash install.sh
```

然后直接运行训练和预测就行

训练过程:

```
python note_model_trainer.py
```

大约八天之后你会得到训练模型，你也可以直接下载作者训练好的模型，然后修改constant.py里的模型名就行。

推理过程:

```
python note_model_test.py
```

大约三天之后你会得到测试结果，运行完之后你会获得三个excel每一行表示每一首歌的评价指标，excel操作算一下平均值即为实验结果

实验二:

```
python limit_note_time
```

运行完之后你会获得res-4、res-5、res-6、res-7、res-8、res-9、res-10这几个excel。res-i表示给定音符可能长度i的结果



如有其它问题发作者邮箱2314879515@qq.com
