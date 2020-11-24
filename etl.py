import psycopg2
import requests
import configparser
import json
import pandas as pd


class etl_handler(object):
    "A class that loads a JSON file, transform data, and insert data into PostgreSQL tables"
    def __init__(self):
        self.config = '/Users/Carl/_data Science/_project/book to podcast/database.ini'
        self.data = None
    
    def psql_insert(self):
        return self.insert_table

    # Loading functions
    def load_json(self, json_file: str, max_line = None):
        """
        Takes a JSON file path and maximum lines to be loaded.
        Saves loaded data in self.data
        """
        data = []
        count = 0
        with open(json_file, 'r') as f:
            for line in f:
                dict_ = json.loads(line)
                data.append(dict_)
                count += 1
                if max_line:
                    if count == max_line:
                        break
            self.data = pd.DataFrame(data)

    # Data wrangling/cleaning functions
    def get_data(self, rows: int = None, random_draw: bool = False, seed: int = None):
        """
        Takes the number of rows to be displayed and whether a randome sample should be drawn.
        Returns a pandas DataFrame with specifications above.
        """
        if self.data is not None:
            if not rows:
                # If 'row' is not given, the all rows from self.data will be read.
                rows = self.data.shape[0]
            if random_draw:
                if seed:
                    return self.data.sample(n=rows, random_state=seed)
            else:
                return self.data[:rows]
        else:
            print("Error. Data not loaded. User load_json() to load data first.")
    
    def select_columns(self, cols: list):
        """Limit the columns of self.data to the columne list provided"""
        self.data = self.data[cols]
    
    def add_column(self, col_name, content):
        """Add a new column to self.data with content provided in *content* """
        self.data[col_name] = content
    
    def filter_value(self, column, value, how = None):
        """Only keeps examples that match the value conditions specified in argument"""
        if how == '==':
            self.data = self.data.loc[self.data[column] == value]

        if how == '>=':
            self.data = self.data.loc[self.data[column] >= value]

        if how == '>':
            self.data = self.data.loc[self.data[column] > value]

        if how == '<=':
            self.data = self.data.loc[self.data[column] <= value]

        if how == '<':
            self.data = self.data.loc[self.data[column] < value]
        
        self.data = self.data.reset_index(drop=True)
        return None
        
    def column_function(self, cols: list, func):
        """Apply function *func* to the specified column in self.data"""
        for col in cols:
            new_col = self.data[col].map(func)
            self.data.loc[:, col] = new_col
        return None
    
    def drop_na(self):
        self.data = self.data.dropna()
        self.data = self.data.reset_index(drop=True)
        return None
    
    # PSQL helper functions
    def psql_connect(self):
        "Given a confige file (config.ini), this function locates information PostgreSQL "
        config = configparser.ConfigParser()
        config.read(self.config)
        connection = psycopg2.connect(
            host = config['postgres']['host'],
            database = config['postgres']['database'],
            user = config['postgres']['user'],
            password = config['postgres']['password']
            )
        cursor = connection.cursor()
        return connection, cursor


class bookgraph_etl(etl_handler):
    def __init__(self):
        super().__init__()
        self.create_table = ("""CREATE TABLE IF NOT EXISTS book(
                                book_id bigint PRIMARY KEY,
                                year smallint NOT NULL,
                                rating_counts int NOT NULL,
                                title varchar(250) NOT NULL UNIQUE,
                                description varchar(500),
                                authors varchar(250),
                                language varchar(20))""")
        
        self.insert_table = ("""INSERT INTO book 
                               (book_id, year, rating_counts, title, description, authors, language)
                               VALUES (%s, %s, %s, %s, %s, %s, %s)
                               ON CONFLICT DO NOTHING""")
        
    def psql_create(self):
        return self.create_table

    def psql_insert(self):
        return self.insert_table


class nyt_etl(etl_handler):
    def __init__(self):
        super().__init__()
        self.create_table = ("""CREATE TABLE IF NOT EXISTS nyt(
                                book_id INT GENERATED ALWAYS AS IDENTITY,
                                book_type varchar(100),
                                bestsellers_date date, 
                                published_date date,
                                weeks_on_list smallint,
                                title varchar(250) NOT NULL,
                                author text,
                                publisher text,
                                description text,
                                primary_isbn13 varchar(13) UNIQUE,
                                primary_isbn10 varchar(10))
                                """)
        
        self.insert_table = ("""INSERT INTO nyt 
                               (book_type, bestsellers_date, published_date, weeks_on_list, title, author, publisher, description, primary_isbn13, primary_isbn10)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                               ON CONFLICT (primary_isbn13) DO UPDATE SET weeks_on_list = EXCLUDED.weeks_on_list""")

        self.config = 'database.ini'
        
    def psql_create(self):
        return self.create_table

    def psql_insert(self):
        return self.insert_table

    def update_bestsellers(self, booktype = 'paperback-nonfiction'):
        """
        self latest NYT Paperpack Non-fiction into self.data
        """
        config = configparser.ConfigParser()
        config.read(self.config)

        request_url = "https://api.nytimes.com/svc/books/v3/lists.json?list={}&api-key={}&offset=0".format(booktype, config['nyt']['api-key'])
        request_headers = {"Accept": "application/json"}

        request = requests.get(request_url, headers=request_headers)
        request = request.json()

        if request['status'] == "OK":
            # a temporary list to hold books
            book_collection = list()

            for book in request['results']:
                # a temporary dictionary to hold data
                d = {}

                # some data are located at the outer-level:
                d['book_type'] = book['list_name']
                d['bestsellers_date'] = book['bestsellers_date']
                d['published_date'] = book['published_date']
                d['weeks_on_list'] = book['weeks_on_list']

                # some data are located at the inner-level under 'book_details':
                inner_columns = ['title', 'author', 'publisher', 'description', 'primary_isbn13', 'primary_isbn10']
                for col in inner_columns:
                  d[col] = book['book_details'][0].get(col)

                book_collection.append(d)

            # set self.data as the newly collected data
            self.data = pd.DataFrame(book_collection)

        else:
            print('Error, connection status was not OK.')
            return None

    def psql_update(self):
        conn, cur = self.psql_connect()
        cur.execute(self.psql_create())
        conn.commit()

        for i, row in self.get_data().iterrows():
            cur.execute(self.psql_insert(), row)
            conn.commit()

        conn.close()






