import requests
import configparser
from etl import nyt_etl

config = configparser.ConfigParser()
config.read('database.ini')

def get_nyt_bestsellers(list_name = 'paperback-nonfiction', iteration = 1):

	# not useable as this particular query only returns 15 books
	def yield_offset():
		offset = 0
		while True:
			yield offset
			offset += 20

	request_url = "https://api.nytimes.com/svc/books/v3/lists.json?list={}&api-key={}&offset=0".format(list_name, config['nyt']['api-key'])
	request_headers = {"Accept": "application/json"}

	request = requests.get(request_url, headers=request_headers)
	request = request.json()

	if request['status'] == "OK":
		# print([request['results'][i]['bestsellers_date'] for i in range(len(request['results']))])
		for i, book in enumerate(request['results']):
			print(book)
			# print(i+1)
			# columns = ['title', 'author', 'publisher', 'description', 'primary_isbn13']
			# for col in columns:
			# 	print(book['book_details'][0].get(col))
			# print('================================================================')
		return None
	else:
		print('Error, connection status was not OK.')
		return None



# ==============

nyt = nyt_etl()
nyt.update_bestsellers()
nyt.psql_update()


