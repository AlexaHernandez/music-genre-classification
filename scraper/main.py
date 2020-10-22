import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

from tqdm import tqdm
import requests
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from sklearn.utils import shuffle
import pandas as pd
import numpy as np

def main():
    website = 'https://www.vagalume.com.br'
    genres = {
        # Other potential canditates with enough data: Indie, maybe pop-rock, maybe reggae, rnb, punk rock, alternative rock, maybe romantica, maybe soul
        # Also to try: Don't disregard an artist immediately if one of their songs is not in English, keep on looking for a couple of songs before discarding.
        'Blues': 'browse/style/blues.html',
        'Rock': 'browse/style/rock.html',
        'Heavy Metal': 'browse/style/heavy-metal.html',
        'Pop': 'browse/style/pop.html',
        'Hip-Hop': 'browse/style/hip-hop.html',
        'Hardcore': 'browse/style/hardcore.html',
        'Hard Rock': 'browse/style/hard-rock.html',
        'Electronica': 'browse/style/electronica.html',
        'Folk': 'browse/style/folk.html',
        'Country': 'browse/style/country.html'
    }

    max_artist_songs = 150
    max_genre_songs = 10000

    data = pd.DataFrame(columns=['artist', 'song', 'lyrics', 'genre'])
    for genre, genre_url in genres.items():
        song_data = sample_songs(website, genre_url, max_artist_songs, max_genre_songs, genre)
        for entry in song_data:
            data = data.append(entry, ignore_index=True)

    logging.info(f'Total number of songs sampled: {len(data)}')

    data.reset_index(drop=True).to_csv('lyrics.csv', index=False)

def sample_songs(website, genre_url, max_artist_songs, max_genre_songs, genre):
    data = []
    random_state = 12345
    
    progress_bar = tqdm(total=max_genre_songs, desc=f'Sampling songs from the genre {genre}')

    n_total = 0
    done = False
    for artist in shuffle(get_artist_urls(urljoin(website, genre_url)), random_state=random_state):
        artist_songs = shuffle(get_song_urls(urljoin(website, artist)), random_state=random_state)
        n_artist_songs = 0
        i = 0
        if done:
            break
        while not done and n_artist_songs < max_artist_songs and i < len(artist_songs):
            song = get_song(urljoin(website, artist_songs[i]))

            # Stop parsing artist if we hit a non English song (speeds up scraping)
            if not song:
                break
            
            song.update({'genre': genre})
            data.append(song)
            
            n_artist_songs += 1
            n_total += 1
            i += 1
            progress_bar.update(1)

            if n_total == max_genre_songs:
                done = True
    
    progress_bar.close()

    return data

def get_artist_urls(url):
    soup = get_soup(url)

    if soup is None:
        return []

    artist_container = soup.findAll('div', {'class': 'moreNamesContainer'})[0]
    return [link['href'] for link in artist_container.find_all('a')]

def get_song_urls(url):
    soup = get_soup(url)

    if soup is None:
        return []

    return [link['href'] for link in soup.find_all('a', {'class': 'nameMusic'})]

def get_song(url, english_only=True):
    soup = get_soup(url)
    
    if soup is None:
        return None

    # If we only want to consider English lyrics
    if english_only and not soup.find('i', {'class': 'lang langBg-eng'}):
        return None

    content = soup.find('div', {'id': 'lyricContent'})

    return {
        'song': content.find('h1').get_text(),
        'artist': content.find('h2').get_text(),
        'lyrics': content.find('div', {'id': 'lyrics'}).get_text('\n')
    }

def get_soup(url):
    try:
        response = requests.get(url)
        return BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        logging.error(e)
        return None

if __name__ == '__main__':
	main()
