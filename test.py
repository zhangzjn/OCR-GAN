from options import Options
from lib.data.dataloader import load_data_FD_aug
from lib.models import load_model

##
def main():
    """ Testing
    """
    opt = Options().parse()
    data = load_data_FD_aug(opt, opt.dataset)
    model = load_model(opt, data, opt.dataset)
    model.test()

if __name__ == '__main__':
    main()
