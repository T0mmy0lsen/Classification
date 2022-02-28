class User:

    _fillables = ['relation_id', 'username', 'email']

    def __eq__(self, other):
        return self.username == other.username \
               and self.email == other.email \
               and self.relation_id == other.relation_id

    def __hash__(self):
        return hash(('username', self.username, 'email', self.email, 'relation_id', self.relation_id))

    def __init__(self, username, email, relation_id):
        self.username = username
        self.email = email
        self.relation_id = relation_id

