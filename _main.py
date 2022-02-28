from classes.ihlp import IHLP
from classes.objs.Request import Request


def main():

    IHLP().get(use_cache=False, use_all=True)

    # import model_ihlp_category
    # import model_ihlp_time

if __name__ == '__main__':
    main()
