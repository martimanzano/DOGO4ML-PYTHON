from TrustAssessor.metrics.Metric import Metric
from uq360.metrics.classification_metrics import expected_calibration_error

class InvertedExpectedCalibrationSKL(Metric):
    """Inverted brier score metric of a sklearn classifier using UQ360.

    Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger; Proceedings of the 34th International Conference on Machine Learning, PMLR 70:1321-1330, 2017.

    Args:
        Metric (Class): Metric interface
    """

    def __init__(self):
        super().__init__()

    def assess(self, trainedModel, dataX, dataY):
        print("TRUST - Computing expected calibration uncertainty metric...")
        prediction = trainedModel.predict(dataX)
        prediction_proba = trainedModel.predict_proba(dataX)
        
        expected_cal_error = expected_calibration_error(dataY, prediction_proba, prediction, len(set(dataY)), False)

        self.assessment = (1-expected_cal_error)
