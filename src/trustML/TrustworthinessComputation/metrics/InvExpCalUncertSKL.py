from trustML.TrustworthinessComputation.metrics.Metric import Metric
from uq360.metrics.classification_metrics import expected_calibration_error

class InvertedExpectedCalibrationSKL(Metric):
    """Inverted brier score metric of a sklearn-based classifier using UQ360.

    This metric measures the difference in expectation between confidence and accuracy. 
    Although it is a cost function, its assessment is inverted so it can be treated as 
    the rest of metrics (i.e., as a percentage).

    Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger; Proceedings of the 34th International Conference on Machine Learning, PMLR 70:1321-1330, 2017.

    ADDITIONAL PROPERTIES:
    None

    Args:
        Metric (Class): Metric abstract class
    """

    def __init__(self):
        super().__init__()

    def assess(self, trainedModel, dataX, dataY):
        print("Computing expected calibration uncertainty metric...")
        prediction = trainedModel.predict(dataX)
        prediction_proba = trainedModel.predict_proba(dataX)
        
        expected_cal_error = expected_calibration_error(dataY, prediction_proba, prediction, len(set(dataY)), False)

        self.value = (1-expected_cal_error)