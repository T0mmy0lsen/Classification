from classes import sql


class Item:

    _fillables = ['id', 'description', 'subject', 'username', 'email', 'sessionid']

    def __init__(self):
        self.id = 0

    @property
    def fillables(self):
        return self._fillables

    @staticmethod
    def get_items():
        query = "SELECT {} FROM `item` WHERE `username` IS NOT NULL".format(
            ', '.join(["`{}`".format(e) for e in Item._fillables]))
        return sql.SQL().all(query, [])

    def get_item(self, item_id):
        query = "SELECT {} FROM `item` WHERE `id` = %s".format(
            ', '.join(["`{}`".format(e) for e in self._fillables]))
        el = sql.SQL().one(query, [item_id])
        for idx, e in enumerate(self._fillables):
            setattr(self, e, el[idx])
        return self