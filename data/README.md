# Data Directory

## ��ȡ���ݼ�

����Ŀʹ��KGAT��ʽ���Ƽ����ݼ���

### ��������

�����������������ݼ���
Amazon-Book: https://github.com/xiangwang1223/knowledge_graph_attention_network/tree/master/Data/amazon-book
Last-FM: https://github.com/xiangwang1223/knowledge_graph_attention_network/tree/master/Data/last-fm
Yelp2018: https://github.com/xiangwang1223/knowledge_graph_attention_network/tree/master/Data/yelp2018

### ���ݽṹ

���غ󣬽����ݷ����ڶ�Ӧ����Ŀ¼�У�

```
data/
������ amazon-book/
��   ������ train.txt
��   ������ test.txt
��   ������ kg_final.txt (��ѡ)
������ last-fm/
��   ������ train.txt
��   ������ test.txt
��   ������ kg_final.txt (��ѡ)
������ yelp2018/
    ������ train.txt
    ������ test.txt
    ������ kg_final.txt (��ѡ)
```

### Ԥ��������

����������������Ƕ���ļ���

```bash
python scripts/prepare_data.py --dataset amazon-book
```

�⽫���ɣ�
`item_embeddings.npy`: ��ƷǶ��
`kg_embeddings.npy`: ֪ʶͼ��Ƕ��

## ע��

Ƕ���ļ���*.npy����ԭʼ�����ļ���*.txt�������ϴ���Git�ֿ⣨�����ӵ�.gitignore����
ÿ���û���Ҫ�Լ��������ݲ�����Ƕ�롣
