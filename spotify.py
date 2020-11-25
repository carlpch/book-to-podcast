#!/Users/Carl/opt/anaconda3/bin/python

from etl import episode_etl
import re

def standardize_description(text, max_len=10_000, min_len = 10):
    text = text.replace('\n',' ')
    text = text.replace("\'",'')
    # Remove web urls    
    text = re.sub('http[s]*://\S+', '', text)
    # Remove phone number
    text = re.sub('[0-9]{4}[0-9]+', '', text)
    # Remove email
    text = re.sub('\S+@\S+', '', text)
    # Remove hashtags
    text = re.sub('#\S+', '', text)
    # Remove repeating space
    text = re.sub('\s\s+', ' ', text)
    text = text.strip()
    
    # rsplit(maxsplit=1)[0] takes away only the right-most part of text separated by '. '
    text = text[:max_len].rsplit('. ', maxsplit=1)[0]
    if len(text) < min_len:
        return None
    else:
        return text



# create connection
podcasts = episode_etl()
podcasts.spotify_connect()

# download data
podcasts.load_show(target_rows = 2000, type="episode", random_search = True, voc_size = 50)

# pre-processing
podcasts.column_function(['description'], standardize_description)
podcasts.drop_na()

# language detection
episode_language = ['other' for i in podcasts.get_data().index]

for i in podcasts.get_data().index:
    try: 
        lang = detect(episode.get_data().description[i])
        episode_language[i] = lang
    except:
        continue

podcasts.add_column('language', episode_language)
podcasts.psql_update()

