# Dataset

We were inspired by this [Kaggle dataset](https://www.kaggle.com/neisse/scrapped-lyrics-from-6-genres). The data comes from the website [Vagalume](https://www.vagalume.com.br) which contains many lyrics from artists across various genres. It contains six genres but unfortunately only three of them: Rock, Pop and Hip-Hop have enough data in English (see notebooks/helgi-kaggle-dataset.ipynb for details). Therefore, we wrote our own scraper to have more genres of English songs. The scraper can be found at scraper/main.py and its resulting datasets downloaded here:
* [scraped-lyrics-v1](https://drive.google.com/file/d/1IxVxRc8DxBBL7jDZJzaMjhWw4rBGqPKg/view?usp=sharing) (10 genre categories, 8000 song lyrics per category)
* [scraped-lyrics-v2](https://drive.google.com/file/d/1GMqCpl2uGhRj7xrz4l40cYYV4U2LwZBS/view?usp=sharing) (Re-run of the scraper with the same parameters as v1 except now multi-genre labels are included per artists' songs)
  * [scraped-lyrics-v2-preprocessed](https://drive.google.com/file/d/11W4I3tqU7bOsbLLHlcJ66fqHXGpygmF9/view?usp=sharing) (By running helgi-05-scraped-lyrics-v2-preprocessing.ipynb)

# Pickled GloVe Embedding Matrices
To save computation time, we loaded the pretrained GloVe embeddings, computed the embedding matrices, and pickled the matrics. The pickled matrices are available [here](https://mcgill-my.sharepoint.com/:f:/g/personal/alexa_hernandez_mail_mcgill_ca/EvjYAKRgi_RGqXaZgevg5LMBnuTr3BBpb2ZD5FfJ_7e82Q?e=saTHKW). This way, when we create our LSTM we directly load the appropriate GloVe embedding matrix.
