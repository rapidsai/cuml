
class RegressorMixin:
    """Mixin class for regression estimators in"""
    _estimator_type = "regressor"

    def score(self, X, y, **kwargs):
        """Scoring function for linear classifiers

        Returns the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : [cudf.DataFrame]
            Test samples on which we predict
        y : [cudf.Series]
            True values for predict(X)

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        from cuml.metrics.regression import r2_score
        return r2_score(y.to_gpu_array(), self.predict(X).to_gpu_array())
