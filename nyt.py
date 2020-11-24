#!/Users/Carl/opt/anaconda3/bin/python

from etl import nyt_etl

# Below are the book types NYT still actively maintains (Manga or graphic books bestsellers has become inactive since 2017 or so. See nyt_list.json for detail)
bestseller_list = ['combined-print-and-e-book-fiction', 'combined-print-and-e-book-nonfiction', 
					'hardcover-fiction', 'hardcover-nonfiction', 'trade-fiction-paperback',
					'mass-market-paperback', 'paperback-nonfiction', 'hardcover-advice', 'paperback-advice', 
					'advice-how-to-and-miscellaneous']

# Just a loop through all the bestseller list and save data into a PostgreSQL table called 'nyt'
for book_type in bestseller_list:
	nyt = nyt_etl()
	nyt.update_bestsellers(booktype = book_type)
	nyt.psql_update()
