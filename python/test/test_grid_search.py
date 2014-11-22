import shutil
import tempfile
import numpy as np

from sklearn.datasets import make_classification

from common import SplearnTestCase
from splearn.rdd import DictRDD
from splearn.grid_search import SparkGridSearchCV
from splearn.naive_bayes import SparkMultinomialNB

from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV

from sklearn.utils.testing import assert_array_almost_equal


class GridSearchTestCase(SplearnTestCase):

    def setUp(self):
        super(GridSearchTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(GridSearchTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)

    def generate_dataset(self, classes, samples, blocks=None):
        X, y = make_classification(n_classes=classes,
                                   n_samples=samples, n_features=4,
                                   random_state=42)
        X = np.abs(X)

        X_rdd = self.sc.parallelize(X)
        y_rdd = self.sc.parallelize(y)
        Z = DictRDD(X_rdd.zip(y_rdd), columns=('X', 'y'), block_size=blocks)

        return X, y, Z


class TestGridSearchCV(GridSearchTestCase):

    def test_same_result(self):
        X, y, Z = self.generate_dataset(2, 30000, 5001)

        parameters = {'alpha': [0.1, 1, 10]}
        fit_params = {'classes': np.unique(y)}

        local_estimator = MultinomialNB()
        local_grid = GridSearchCV(estimator=local_estimator,
                                  param_grid=parameters)

        estimator = SparkMultinomialNB()
        grid = SparkGridSearchCV(estimator=estimator,
                                 param_grid=parameters,
                                 fit_params=fit_params)

        local_grid.fit(X, y)
        grid.fit(Z)

        locscores = [r.mean_validation_score for r in local_grid.grid_scores_]
        scores = [r.mean_validation_score for r in grid.grid_scores_]

        assert_array_almost_equal(locscores, scores, decimal=2)
