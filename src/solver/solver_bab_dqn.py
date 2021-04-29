from src.solver.solver_tsptw import solve as cppsolve
from src.problem.tsptw.environment.tsptw import TSPTW

N_NODES_LIST = [20, 50, 100]


def solve(instance: TSPTW, time_limit: int = 6, n_nodes_trained: int = None) -> list:
    if n_nodes_trained is None:
        for node in N_NODES_LIST:
            if node - instance.n_city > 0:
                n_nodes_trained = node
                break
        else:
            n_nodes_trained = N_NODES_LIST[-1]
    solution = cppsolve(instance, time_limit, n_nodes_trained)
    if len(solution) > 0:
        return [0] + solution + [0]
    else:
        return []
