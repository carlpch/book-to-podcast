
# Demo code sample. Not indended for production use.

# See instructions for installing Requests module for Python
# http://docs.python-requests.org/en/master/user/install/

import requests
import configparser

config = configparser.ConfigParser()
config.read('database.ini')



def get_nyt_bestsellers(list_name = 'paperback-nonfiction', iteration = 1):

    def offset():
        offset = 0
        while True:
            yield offset
            offset += 20

    requestUrl = "https://api.nytimes.com/svc/books/v3/lists.json?list={}&api-key={}&offset=0".format(list_name, config['nyt']['api-key'])
    requestHeaders = {

    "Accept": "application/json"
    }

    request = requests.get(requestUrl, headers=requestHeaders)
    request = request.json()

    if request['status'] == "OK":
        return request['results']
    else:
        print('Error, connection status was not OK.')
        return None


if __name__ == "__main__":
    get_nyt_bestsellers()




def 



