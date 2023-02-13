class Metric:
    """
    Metric abstract class
    """

    def __init__(self):
        self.assessment = None
    
    def assess(self, trainedModel, dataX, dataY):
        """Assessment of the metric using the trained model, dataset predictors and targets passed as parameters.
        """
        
        # self.assessment =
        pass
        
    
    def getAssessment(self) -> float:
        return self.assessment