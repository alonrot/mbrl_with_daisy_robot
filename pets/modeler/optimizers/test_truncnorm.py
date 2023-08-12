# This will visualize the two distributions, enable if you want to validate this manually.
# This is not an automatic unit test.

# import matplotlib.pyplot as plt
# import torch
# from scipy.stats import truncnorm
#
# from mbrl import utils
#
# fig, ax = plt.subplots(1, 1)
#
# def test_truncnorm():
#     a, b = -2, 2
#     size = 1000000
#     r = truncnorm.rvs(a, b, size=size)
#     ax.hist(r, density=True, histtype='stepfilled', alpha=0.2, bins=50)
#
#     tensor = torch.zeros(size)
#     utils.truncated_normal_(tensor)
#     r = tensor.numpy()
#
#     ax.hist(r, density=True, histtype='stepfilled', alpha=0.2, bins=50)
#     ax.legend(loc='best', frameon=False)
#     plt.show()
#
#
# if __name__ == '__main__':
#     test_truncnorm()
