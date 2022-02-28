from classes.objs.Relation import Relation
from classes.objs.Request import Request


class Meta:

    solver_count_threshold = 1
    solvers_max_len = 0
    solvers_dict = {}
    solvers_dict_count = {}
    solvers_dict_inverse = {}
    solvers = []

    post_solvers = []
    post_solvers_dict_count = {}

    def buildDict(self, df):

        # Initiate dict for count of solvers
        for idx, solver in enumerate(self.post_solvers):
            self.post_solvers_dict_count[solver] = 0

        # Count all the solvers
        for idx, row in df.iterrows():
            for el in enumerate(row.solvers):
                self.post_solvers_dict_count[el] = self.post_solvers_dict_count[el] + 1

        # Initiate all solvers over a given threshold
        for idx, solver in enumerate(self.post_solvers):
            if self.post_solvers_dict_count[solver] > self.solver_count_threshold:
                self.solvers.append(solver)
                self.solvers_dict[idx] = solver
                self.solvers_dict_count[idx] = 0
                self.solvers_dict_inverse[solver] = idx

    def print(self):

        print('Count:')
        print(self.solvers_dict_count)
        print('Mapping:')
        print(self.solvers_dict)


def findSolvers(m):

    df_requests = Request.get_sql(10)
    df_relations = Relation.get_relations_by_type('ItemRole')

    df_requests['solvers'] = df_requests.apply(lambda x: determineSolvers(x, df_relations, meta=m), axis=1)
    df_requests = df_requests[len(df_requests.solvers) > 0]

    # No need to pad them at this point. We wait with the vector until later.
    # df_requests['solvers'] = df_requests.apply(lambda x: padSolvers(x, meta=m), axis=1)

    # The label will be of type [0, 0, 0, 1] which tells us that key = 3 in dict is the actual solver.
    # If label = [1, 0, 0, 1], the both key 0 and 3 solved the Request.
    m.buildDict(df_requests)

    df_requests['solvers'] = df_requests.apply(lambda x: buildLabel(x, meta=m), axis=1)

    m.print()
    print(df_requests)


def buildLabel(x, meta):

    label = []
    for idx, solver in enumerate(meta.solvers):
        if meta.solvers_dict[idx] in x.solvers:
            label.append(1)
            meta.solvers_dict_count[idx] = meta.solvers_dict_count[idx] + 1
        else:
            label.append(0)
    return label


def padSolvers(x, meta):

    tmp = []
    tmp = tmp + [0] * (meta.solvers_max_len - len(x.solvers))
    tmp = tmp + x.solvers
    return tmp


def determineSolvers(x, relations, meta):

    tmp = relations[relations.leftId == x['id']]
    tmp = tmp[tmp.rightType == 'ItemRole']

    if len(tmp) > 2:

        # Sort by id since we expect the ID's sequence to match the ordering of when the ItemRole was added to the Request.
        # We say that id = 100 was assign a request after id = 99, etc.
        tmp.sort_values(by=['id'], inplace=True)

        # We expect the top two to be the requester and the dispatcher.
        rm = tmp.iloc[0:2]

        # We remove them from the set of ItemRoles. We expect these two to be redundant in solving the issue.
        rm_list = rm.rightId.tolist()
        tmp = tmp[~tmp.rightId.isin(rm_list)]

        # Remove duplicates for now, to reduce complexity.
        tmp_no_duplicates = list(dict.fromkeys(tmp.rightId.tolist()))

        # Max length will set the length of the label-vector.
        if len(tmp_no_duplicates) > meta.solvers_max_len:
            meta.solvers_max_len = len(tmp_no_duplicates)

        # Build a list of solvers so we can get length and indexes for each.
        for solver in tmp_no_duplicates:
            if solver not in meta.post_solvers:
                meta.post_solvers.append(solver)

        return tmp_no_duplicates
    else:
        return []


def main():
    m = Meta()
    findSolvers(m)


if __name__ == '__main__':
    main()
