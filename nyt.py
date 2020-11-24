from etl import nyt_etl

nyt = nyt_etl()
nyt.update_bestsellers()
nyt.psql_update()


