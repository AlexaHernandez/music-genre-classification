import sys
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

from tqdm import tqdm
import requests
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from sklearn.utils import shuffle
import pandas as pd

def main():
    website = 'https://www.vagalume.com.br'
    genres = {
        'Alternative Rock': 'browse/style/rock-alternativo.html',
        'Country': 'browse/style/country.html',
        'Hard Rock': 'browse/style/hard-rock.html',
        'Heavy Metal': 'browse/style/heavy-metal.html',
        'Hip-Hop': 'browse/style/hip-hop.html',
        'Indie': 'browse/style/indie.html',
        'Pop': 'browse/style/pop.html',
        'R&B': 'browse/style/r-n-b.html',
        'Rock': 'browse/style/rock.html',
        'Soul': 'browse/style/soul-music.html'
    }

    max_genre_songs = 8000
    max_artist_songs = None

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

    max_artist_songs = sys.maxsize if max_artist_songs is None else max_artist_songs
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

            # Skip artist if a song is not in English, speeds up scraping
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
