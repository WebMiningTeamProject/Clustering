import logging
import sys
import warnings
import threading

import MySQLdb
import MySQLdb.cursors

###BASED ON Seb's DatabaseHandler

class DatabaseHandler:
    def __init__(self, host, user, password, db_name):
        self.host = host
        self.user = user
        self.password = password
        self.db_name = db_name
        self.logger = logging.getLogger()
        self.cnx = None
        self.db = None
        self.connect()
        self.condition = threading.Condition()

    def connect(self):
        try:
            self.db = MySQLdb.connect(
                self.host, self.user, self.password, self.db_name, cursorclass=MySQLdb.cursors.DictCursor, charset='utf8')

            warnings.filterwarnings("error", category=MySQLdb.Warning)
            self.cnx = self.db.cursor()
            self.db.autocommit(True)
            self.logger.debug(
                'Connected to database %s on %s with user %s' %
                (self.db_name, self.host, self.user))
        except MySQLdb.Error as e:
            self.logger.error(
                "Error while establishing connection to the database server [%d]: %s"
                % (e.args[0], e.args[1]))
            sys.exit(1)

    def close(self):
        self.cnx.close()
        self.db.close()
        self.logger.debug('Closed DB connection')



    def __buildInsertSql(self, table, objs):
        if len(objs) == 0:
            return None
        s = set()
        [s.update(row.keys()) for row in objs]
        columns = [col for col in s]
        tuples = []
        for item in objs:
            if item:
                values = []
                for key in columns:
                    try:
                        values.append('"%s"' % str(item[key]).replace('"', "") if not item[key] == '' else 'NULL')
                    except KeyError:
                        values.append('NULL')
                if not all('NULL' == value for value in values):
                    tuples.append('(%s)' % ', '.join(values))
        return 'INSERT INTO `' + table + '` (' + ', '.join(
            ['`%s`' % column for column in columns]) \
            + ') VALUES\n' + ',\n'.join(tuples)


    ##Build select satement without constraints
    def __buildSelectSql(self,tableName):
        statement = 'SELECT * FROM`' + tableName + ';'
        return statement;

    #Execute SQL-statement
    def execute(self, statement):
        if statement:
            with self.condition:
                try:
                    self.logger.debug('Executing SQL-query:\n\t%s'
                                      % statement.replace('\n', '\n\t'))
                    self.cnx.execute(statement)
                    return self.cnx.fetchall()
                except MySQLdb.Warning as e:
                    self.logger.warn("Warning while executing statement: %s" % e)
                except MySQLdb.Error as e:
                    self.logger.error("Error while executing statement [%d]: %s"
                                      % (e.args[0], e.args[1]))

    def persistDict(self, table, array_of_dicts):
        sql = self.__buildInsertSql(table, array_of_dicts)
        self.execute(sql)

    def select(self, table):
        sql = self.__buildSelectSql(table)
        resultSet = self.execute(sql)
        return resultSet









