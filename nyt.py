#!/Users/Carl/opt/anaconda3/bin/python

import time
from etl import nyt_etl

# Below are the book types NYT still actively maintains (Manga or graphic books bestsellers has become inactive since 2017 or so. See nyt_list.json for detail)
bestseller_list = ['combined-print-and-e-book-fiction', 'combined-print-and-e-book-nonfiction', 
					'hardcover-fiction', 'hardcover-nonfiction', 'trade-fiction-paperback',
					'mass-market-paperback', 'paperback-nonfiction', 'hardcover-advice', 'paperback-advice', 
					'advice-how-to-and-miscellaneous', 
					'young-adult-hardcover', 
					'business-books', 
					'graphic-books-and-manga', 'mass-market-monthly', 'middle-grade-paperback-monthly', 'young-adult-paperback-monthly']

# Just a loop through all the bestseller list and save data into a PostgreSQL table called 'nyt'
for book_type in bestseller_list:
	print('Starting process for {}'.format(book_type))
	nyt = nyt_etl()
	nyt.update_bestsellers(booktype = book_type)
	nyt.psql_update()

	# https://developer.nytimes.com/faq
	# 11. Is there an API call limit?
	# Yes, there are two rate limits per API: 4,000 requests per day and 10 requests per minute. 
	# You should sleep 6 seconds between calls to avoid hitting the per minute rate limit. If you need a higher rate limit, please contact us at code@nytimes.com.
	time.sleep(6)
