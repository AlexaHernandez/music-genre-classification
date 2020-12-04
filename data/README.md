# Dataset

We were inspired by this [Kaggle dataset](https://www.kaggle.com/neisse/scrapped-lyrics-from-6-genres). The data comes from the website [Vagalume](https://www.vagalume.com.br) which contains many lyrics from artists across various genres. It contains six genres but unfortunately only three of them: Rock, Pop and Hip-Hop have enough data in English (see notebooks/helgi-kaggle-dataset.ipynb for details). Therefore, we wrote our own scraper to have more genres of English songs. The scraper can be found at scraper/main.py and its resulting datasets downloaded here:
* [Version 1](https://drive.google.com/file/d/1IxVxRc8DxBBL7jDZJzaMjhWw4rBGqPKg/view?usp=sharing) (10 genres, 8000 song lyrics per genre)
* [Version 2](https://drive.google.com/file/d/1GMqCpl2uGhRj7xrz4l40cYYV4U2LwZBS/view?usp=sharing) (Similar to version 1 except all bands' genres are included)
  * [Preprocessed](https://drive.google.com/file/d/11W4I3tqU7bOsbLLHlcJ66fqHXGpygmF9/view?usp=sharing) (By running helgi-05-scraped-lyrics-v2-preprocessing.ipynb)
