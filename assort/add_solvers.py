from classes.objs.Relation import Relation
from classes.objs.Request import Request
import pandas as pd

pd.options.mode.chained_assignment = None


class Meta:

    solver_count_threshold = 200
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
            for el in row.solversPadded:
                # Ignore the padding
                if el != 0:
                    self.post_solvers_dict_count[el] = self.post_solvers_dict_count[el] + 1

        # Initiate all solvers over a given threshold
        idx = 0
        for solver in self.post_solvers:
            if self.post_solvers_dict_count[solver] > self.solver_count_threshold:
                self.solvers.append(solver)
                self.solvers_dict[idx] = solver
                self.solvers_dict_count[idx] = 0
                self.solvers_dict_inverse[solver] = idx
                idx = idx + 1

    def print(self):
        print('Count:')
        print(self.solvers_dict_count)
        print('Mapping:')
        print(self.solvers_dict)


def findSolvers(m, df):

    df_requests = df  # Request.get_sql(1000)
    df_relations = Relation.get_relations_by_type('ItemRole')

    # The initial solvers are set
    df_requests['solversDetermine'] = df_requests.apply(lambda x: determineSolvers(x, df_relations, meta=m), axis=1)
    df_requests = df_requests[df_requests['solversDetermine'].map(len) > 0]

    # Padded solvers are used whn the label is a vector, i.e. when we care about the order of solvers.
    df_requests['solversPadded'] = df_requests.apply(lambda x: padSolvers(x, meta=m), axis=1)

    # Build dict of solvers and count. Goal is to have data to filter out lines that has uncommon solvers.
    m.buildDict(df_requests)

    # Given dimension = 1; the label will be of type [0, 0, 0, 1] which tells us that key = 3 in dict is the actual solver.
    # If label = [1, 0, 0, 1], the both key 0 and 3 solved the Request.
    # Given dimension = 2; the label will be of type [[1, 0, 1], [0, 1, 0], [0, 0, 0]] tells us we have 3 solvers in the dict.
    # The request was solved by going from solver = 0, then solver = 1, then back to solver = 0.
    df_requests['solvers'] = df_requests.apply(lambda x: buildLabel(x, meta=m, dimensions=1), axis=1)

    # We might want labels with [0, 0, 0] since we would state Requests to be too 'complex'.
    df_requests = df_requests[df_requests['solvers'].map(sum) > 0]

    # m.print()
    # print(len(df_requests))
    return df_requests


def buildLabel(x, meta, dimensions=1):

    # If we use a single dimension we don't care about the order in which the solvers solved the Request.
    if dimensions == 1:
        label = []
        for idx, solver in enumerate(meta.solvers):
            if meta.solvers_dict[idx] in x.solversPadded:
                label.append(1)
                meta.solvers_dict_count[idx] = meta.solvers_dict_count[idx] + 1
            else:
                label.append(0)
        return label

    if dimensions == 2:
        print('Not implemented.')


def padSolvers(x, meta):

    # Pad solvers s.t. all solver vectors are same length.
    # Padded solvers has no effect when the label has one dimension.
    tmp = []
    tmp = tmp + [0] * (meta.solvers_max_len - len(x.solversDetermine))
    tmp = tmp + x.solversDetermine
    return tmp


def determineSolvers(x, relations, meta):

    tmp = relations[relations.leftId == x['id']]
    tmp = tmp[tmp.rightType == 'ItemRole']

    # If there is a solver there must be more ItemRole than just the requester and the dispatcher.
    if len(tmp) > 2:

        # Sort by id since we expect the id's sequence to match the ordering of when the ItemRole was added to the Request.
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


def setSolver(df):
    m = Meta()
    return findSolvers(m, df)
