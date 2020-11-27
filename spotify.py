#!/Users/Carl/opt/anaconda3/bin/python

from etl import episode_etl

# create connection
podcasts = episode_etl()
podcasts.spotify_connect()

# download data
podcasts.load_show(target_rows = 10000, type="episode", random_search = True, voc_size = 50)

# pre-processing
podcasts.psql_update()

