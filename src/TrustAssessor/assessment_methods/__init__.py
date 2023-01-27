import os
import importlib
pyfile_extes = ['py', ]
__all__ = [importlib.import_module('.%s' % filename, __package__) for filename in [os.path.splitext(i)[0] for i in os.listdir(os.path.dirname(__file__)) if os.path.splitext(i)[1] in pyfile_extes] if not filename.startswith('__')]
del os, importlib, pyfile_extes

from TrustAssessor.assessment_methods.WeightedAverage import WeightedAverage
#import TrustAssessor.assessment_methods.BayesianNetwork
from TrustAssessor.assessment_methods.BayesianNetwork import BayesianNetwork