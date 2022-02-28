import re
import datetime
import pandas as pd

from classes import sql
from bs4 import BeautifulSoup

from classes.objs.Relation import Relation


class Request:

    _fillables = ['id', 'description', 'subject', 'solution', 'receivedDate', 'solutionDate', 'deadline', 'priority']
    _relations = []
    _users = []

    @staticmethod
    def get_has_error(x):
        return (
            isinstance(x['solutionDate'], str)
        )

    @staticmethod
    def get_process(x):
        return int(x.solutionDate.timestamp()) - int(x.receivedDate.timestamp())

    @staticmethod
    def get_cleanup(x):
        text = BeautifulSoup(x.description, "lxml").text
        text = text.lower()
        text = re.sub('[\n.]', ' ', text)
        return text

    @property
    def fillables(self):
        return self._fillables

    @property
    def users(self):
        return self._users

    @property
    def relations(self):
        return self._relations

    @property
    def weight(self):
        return int(getattr(self, 'priority'))

    @property
    def deadline_timestamp(self):
        return getattr(self, 'deadline').timestamp()

    def __init__(self):
        self.id = 0

    def get_request(self, request_id):
        query = "SELECT {} FROM `request` WHERE `id` = %s".format(
            ', '.join(["`{}`".format(e) for e in self._fillables]))
        el = sql.SQL().one(query, [request_id])
        for idx, e in enumerate(self._fillables):
            setattr(self, e, el[idx])
        return self

    def get_relations(self):
        self._relations = Relation.get_relations(self.id)
        return self

    def get_items(self):
        for el in self.relations:
            el.get_right()
        return self

    @staticmethod
    def get_sql(test=False):
        sql_test = ''
        if test is not False:
            str_from = str(datetime.datetime(2016, 3, 15))
            sql_test = " WHERE `receivedDate` >= '{}' LIMIT {}".format(str_from, test)
        query = "SELECT {} FROM `request`{}".format(
            ', '.join(["`{}`".format(e) for e in Request._fillables]), sql_test)
        result = sql.SQL().all(query, [])
        return pd.DataFrame(result, columns=Request().fillables)

    @staticmethod
    def get_between_sql(t1, t2):
        query = "SELECT {} FROM `request` WHERE receivedDate >= %s and receivedDate < %s".format(
            ', '.join(["`{}`".format(e) for e in Request._fillables]))
        result = sql.SQL().all(query, [t1, t2])
        return pd.DataFrame(result, columns=Request().fillables)
