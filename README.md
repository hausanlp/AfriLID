# AfriLID

Language identification for African languages, with a focus on social media text.

**Authors:** Idris Abdulmumin, Shamsuddeen Hassan Muhammad and Ibrahim Said Ahmad ([HausaNLP](https://github.com/hausanlp))

AfriLID releases two models on Hugging Face. Both are XLM-RoBERTa classifiers built on [AfroXLMR](https://huggingface.co/Davlan/afro-xlmr-large), a version of XLM-R adapted to African languages.

| Model | Labels | Base model | Test accuracy | Macro F1 | Best for |
|:------|:------:|:-----------|:-------------:|:--------:|:---------|
| [HausaNLP/AfriSentiLID-115](https://huggingface.co/HausaNLP/AfriSentiLID-115) | 116 | AfroXLMR-base | 0.956 | 0.892 | General use, unknown or mixed input |
| [HausaNLP/AfriSentiLID](https://huggingface.co/HausaNLP/AfriSentiLID) | 12 | AfroXLMR-large | 0.990 | 0.989 | Tweets known to be in one of the 12 AfriSenti languages |

The two scores come from different test sets, so do not compare the models' numbers directly.

## Quick start

```bash
pip install transformers torch
```

```python
from transformers import pipeline

lid = pipeline("text-classification", model="HausaNLP/AfriSentiLID-115", top_k=3)

for preds in lid(["Yara sun tafi makaranta da safe, amma malaminsu bai zo ba saboda ruwan sama.",
                  "Gọọmentị etiti ekwuputala na a ga-amalite ịkwụ ụgwọ ọnwa ọhụrụ n'ọnwa na-abịa."],
                 truncation=True, max_length=128):
    print([(p["label"], round(p["score"], 3)) for p in preds])
```

Labels are ISO 639-3 codes. To show language names, install `iso639-lang` and run:

```python
from iso639 import Lang
Lang("hau").name   # 'Hausa'
```

### Label a CSV file

```python
import pandas as pd

df = pd.read_csv("data.csv")              # a column named "text"
preds = lid(df["text"].tolist(), truncation=True, max_length=128)
for k in range(3):
    df[f"lang{k + 1}"] = [p[k]["label"] for p in preds]
    df[f"score{k + 1}"] = [p[k]["score"] for p in preds]
df.to_csv("data_with_lang.csv", index=False)
```

## Languages

**AfriSentiLID-115** covers 116 labels: 108 African languages and varieties, plus Arabic varieties, French, Portuguese, Hebrew and Maltese. The model card has the [full list](https://huggingface.co/HausaNLP/AfriSentiLID-115#languages) and per-language scores.

**AfriSentiLID** covers the 12 AfriSenti languages: Amharic, Algerian Arabic, Moroccan Arabic, Hausa, Igbo, Kinyarwanda, Nigerian Pidgin, Mozambican Portuguese, Swahili, Xitsonga, Twi and Yorùbá.

## Limitations

- Each model always returns one of its own labels, even for unsupported languages; a low score is a hint that the text may not be covered.
- AfriSentiLID-115 never correctly predicts nine labels with little training data (`acm`, `afb`, `aka`, `apc`, `bam`, `fuc`, `fuv`, `rif`, `swh`). It also confuses close relatives such as the Arabic varieties and the Swahili labels. See the model card for details.
- Standard Swahili is often labelled `swc`, and Yorùbá without tone marks can be labelled Gun (`guw`).
- Code-switched text gets a single label, and one- or two-word inputs are unreliable.

## License

The models are released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); see [LICENSE](LICENSE).

## Citation

```bibtex
@misc{abdulmumin2023afrilid,
  title  = {AfriLID: Language Identification for African Languages},
  author = {Abdulmumin, Idris and Muhammad, Shamsuddeen Hassan and Ahmad, Ibrahim Said},
  year   = {2023},
  publisher = {HausaNLP},
  howpublished = {\url{https://github.com/hausanlp/AfriLID}}
}
```

Please also cite the AfriSenti and AfriHate papers:

```bibtex
@inproceedings{muhammad-etal-2023-afrisenti,
    title = "{A}fri{S}enti: A {T}witter Sentiment Analysis Benchmark for {A}frican Languages",
    author = "Muhammad, Shamsuddeen Hassan  and
      Abdulmumin, Idris  and
      Ayele, Abinew Ali  and
      Ousidhoum, Nedjma  and
      Adelani, David Ifeoluwa  and
      Yimam, Seid Muhie  and
      Ahmad, Ibrahim Sa'id  and
      Beloucif, Meriem  and
      Mohammad, Saif M.  and
      Ruder, Sebastian  and
      Hourrane, Oumaima  and
      Brazdil, Pavel  and
      Jorge, Alipio  and
      Ali, Felermino D{\'a}rio M{\'a}rio Ant{\'o}nio  and
      David, Davis  and
      Osei, Salomey  and
      Shehu Bello, Bello  and
      Ibrahim, Falalu  and
      Gwadabe, Tajuddeen  and
      Rutunda, Samuel  and
      Belay, Tadesse  and
      Messelle, Wendimu Baye  and
      Balcha, Hailu Beshada  and
      Chala, Sisay Adugna  and
      Gebremichael, Hagos Tesfahun  and
      Opoku, Bernard  and
      Arthur, Stephen",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.862/",
    doi = "10.18653/v1/2023.emnlp-main.862",
    pages = "13968--13981"
}

@inproceedings{muhammad-etal-2025-afrihate,
    title = "{A}fri{H}ate: A Multilingual Collection of Hate Speech and Abusive Language Datasets for {A}frican Languages",
    author = {Muhammad, Shamsuddeen Hassan  and
      Abdulmumin, Idris  and
      Ayele, Abinew Ali  and
      Adelani, David Ifeoluwa  and
      Ahmad, Ibrahim Said  and
      Aliyu, Saminu Mohammad  and
      R{\"o}ttger, Paul  and
      Oppong, Abigail  and
      Bukula, Andiswa  and
      Chukwuneke, Chiamaka Ijeoma  and
      Jibril, Ebrahim Chekol  and
      Ismail, Elyas Abdi  and
      Alemneh, Esubalew  and
      Gebremichael, Hagos Tesfahun  and
      Aliyu, Lukman Jibril  and
      Beloucif, Meriem  and
      Hourrane, Oumaima  and
      Mabuya, Rooweither  and
      Osei, Salomey  and
      Rutunda, Samuel  and
      Belay, Tadesse Destaw  and
      Guge, Tadesse Kebede  and
      Asfaw, Tesfa Tegegne  and
      Wanzare, Lilian Diana Awuor  and
      Onyango, Nelson Odhiambo  and
      Yimam, Seid Muhie  and
      Ousidhoum, Nedjma},
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the {A}mericas Chapter of the {A}ssociation for {C}omputational {L}inguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.92/",
    doi = "10.18653/v1/2025.naacl-long.92",
    pages = "1854--1871",
    ISBN = "979-8-89176-189-6"
}
```
