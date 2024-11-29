# inference 실행

```bash
./inference.sh
```

# 프로젝트 구조
```bash
level2-nlp-generationfornlp-nlp-11-lv3
├── data
│   ├── train.csv
│   ├── test.csv
│   ├── processed_wiki_ko.txt
│   ├── data_utils.py
│   └── view_and_label_data.ipynb
├── db             
│   ├── bm25_retriever
│   │   └── bm25.bin
│   └── vectorstore
│       ├── index.faiss
│       └── index.pkl
├── source                
│   ├── inference.py
│   ├── fine_tuning.py
│   ├── rag.py
│   ├── vllm_infer.ipynb
│   └── utils.py
│        
├── README.md
├── inference.sh
└── requirements.txt
```
