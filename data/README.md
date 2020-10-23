# Dataset

We were inspired by this [Kaggle dataset](https://www.kaggle.com/neisse/scrapped-lyrics-from-6-genres). The data comes from the website [Vagalume](https://www.vagalume.com.br) which contains many lyrics from artists across various genres. It contains six genres but unfortunately only three of them: Rock, Pop and Hip-Hop have enough data in English (see notebooks/helgi-kaggle-dataset.ipynb for details). Therefore, we wrote our own scrapper to have more genres of English songs. The scraper can be found at scraper/main.py and its resulting datasets downloaded here:
* [Version 1](https://drive.google.com/file/d/1f9DwuW3pvXtRuFPgxkLw5sXO0fL5bLko/view?usp=sharing) (Blues, Rock, Heavy Metal, Pop, Hip-Hop, Hardcore, Hard Rock, Electronica, Folk and Country)
  * See exploration notebook: notebooks/helgi-scrapped-lyrics-v1.ipynb
