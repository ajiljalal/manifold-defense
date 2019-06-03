import torch as ch
from scipy.misc import imsave
import sys

x = ch.load("intermediates_success/batch_%d_orig" % (int(sys.argv[1]),))
imsave("test_orig.png", x.squeeze().cpu().numpy())
x = ch.load("intermediates_success/batch_%d_attack" %  (int(sys.argv[1]),))
imsave("test_attack.png", x.squeeze().view(28,28).cpu().numpy())

