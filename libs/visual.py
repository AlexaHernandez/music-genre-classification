
def analyse_lyrics(dataframe, n_samples, lyrics_length, mode='less', random_state=None, max_print=50):
    """Analyse lyrics

    Args:
        dataframe (pandas.DataFrame): A DataFrame containing lyrics
        n_samples (int): Num samples to print on screen
        lyrics_length (int): How long the lyrics need to be to be considered (see mode)
        mode (str, optional): If 'less', analyses lyrics of length lyrics_lengths or less. Otherwise, if 'more', of lyrics_lenghts or more.
        random_state (int, optional): The random seed for sampling.
        max_print (int, optional): Maximum characters to print for each sampled song's lyrics.

    Raises:
        ValueError: If mode is not 'less' or 'more'
    """
    if mode.lower() == 'less':
        candidates = dataframe[dataframe.lyrics.str.len() <= lyrics_length]
    elif mode.lower() == 'more':
        candidates = dataframe[dataframe.lyrics.str.len() >= lyrics_length]
    else:
        raise ValueError('mode can only be "less" or "more"')

    print(f'There are {len(candidates)} songs with lyrics of {lyrics_length} characters or {mode}.')
    print(f'Here are {n_samples} samples:', end='\n\n')

    for index, entry in candidates.sample(n_samples, random_state=random_state).iterrows():
        lyrics = entry.lyrics
        if len(lyrics) >= max_print:
            lyrics = lyrics[:max_print] + f' <... {len(lyrics) - max_print} more chars>'
        print(f'<index: {index}>\n{lyrics}', end='\n\n')
