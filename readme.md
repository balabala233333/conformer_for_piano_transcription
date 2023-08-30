本github当中主要用于复现论文当中的实验数据,本文已将所有的实验结果excel放在此github中:

想要运行您需要一根tesla A40，并且下载对应的数据集:

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

推理过程:

```
python note_model_test.py
```

运行完之后你会获得三个excel每一行表示每一首歌的评价指标，excel操作算一下平均值即为实验结果

实验二:

```
python limit_note_time
```

运行完之后你会获得
