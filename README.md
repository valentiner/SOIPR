##  SOIRP: Subject-Object Interaction and Reasoning Path Based Joint Relational Triple Extraction by Table Filling.


## Requirements
The dependency packages can be installed with the command:
```
pip install -r requirements.txt
```
## Datasets  
  
- [NYT*](https://github.com/weizhepei/CasRel/tree/master/data/NYT) and [WebNLG*](https://github.com/weizhepei/CasRel/tree/master/data/WebNLG)(following [CasRel](https://github.com/weizhepei/CasRel))  
- [NYT](https://drive.google.com/file/d/1kAVwR051gjfKn3p6oKc7CzNT9g2Cjy6N/view)(following [CopyRE](https://github.com/xiangrongzeng/copy_re))  
- [WebNLG](https://github.com/yubowen-ph/JointER/tree/master/dataset/WebNLG/data)(following [ETL-span](https://github.com/yubowen-ph/JointER))  
  
Or you can just download our preprocessed [datasets](https://drive.google.com/file/d/1ySRSN0EkQ4Qdi-VIQnwfJwofV6AJe9Bs/view?usp=drive_link).

## Usage
* **Get pre-trained BERT model**
Download [BERT-BASE-CASED](https://huggingface.co/bert-base-cased) and put it under `./pretrained`.

* **Train and select the model**
```
python run.py --dataset=WebNLG  --train=train
python run.py --dataset=WebNLG_star   --train=train
python run.py --dataset=NYT   --train=train
python run.py --dataset=NYT_star   --train=train
```

* **Evaluate on the test set**
```
python run.py --dataset=WebNLG  --train=test
python run.py --dataset=WebNLG_star   --train=test
python run.py --dataset=NYT   --train=test
python run.py --dataset=NYT_star   --train=test
```
## Main results
![](./Main_results.png)
The average results of our model throughout the three runs with various random seeds are reported here to demonstrate the reliability and generalizability of our model.

### Acknowledgement
Parts of our codes come from [bert4keras](https://github.com/bojone/bert4keras).
