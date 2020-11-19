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








