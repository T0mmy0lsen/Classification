from classes import sql
from classes.objs.Item import Item
from classes.objs.Object import Object

import pandas as pd


class Relation:

    _fillables = ['id', 'leftId', 'rightId', 'leftType', 'rightType']

    def __init__(self):
        self.id = 0

    @property
    def fillables(self):
        return self._fillables

    @staticmethod
    def get_relations_by_type(rightType):
        query = "SELECT {} FROM `relation` WHERE `rightType` = %s ORDER BY `id` DESC".format(
            ', '.join(["`{}`".format(e) for e in Relation._fillables]))
        result = sql.SQL().all(query, [rightType])
        return pd.DataFrame(result, columns=Relation().fillables)

    @staticmethod
    def get_by_left_id(leftId):
        query = "SELECT {} FROM `relation` WHERE `leftId` = %s".format(
            ', '.join(["`{}`".format(e) for e in Relation._fillables]))
        result = sql.SQL().all(query, [leftId])
        return pd.DataFrame(result, columns=Relation().fillables)