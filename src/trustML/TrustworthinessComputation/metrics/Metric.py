class Metric:
    """
    Metric abstract class
    """
      
    def __init__(self):
        self.value = None
    
    def assess(self, trainedModel, dataX, dataY):
        """Assessment of the metric using the trained model, dataset predictors and targets passed as parameters.
        """

        pass
    
    def getValue(self) -> float:
        return self.value