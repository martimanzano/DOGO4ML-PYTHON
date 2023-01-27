class AssessmentMethod:
    """Assessment method interface
    """
    def __init__(self):
        """Initializes and loads parameters from the trustable entity and its configuration dict"""
        self.assessment = None
        self.assessmentObject = None
        self.metrics = []

    def assess(self):
        """Assesses the trustable entity using the assessment method (child class) and stores the resulting assessment
        """
        pass
    
    def getAssessment(self) -> str:
        """Returns the computed assessment in a user-friendly format
        """
        return self.assessment

    def getAssessmentObject(self):
        """Returns the computed assessment in a user-friendly format
        """
        return self.assessmentObject

    def getMetricsAssessmentDict(self) -> dict:
        metricNames = [metric.__class__.__name__ for metric in self.metrics]
        metricAssessments = [metric.getAssessment() for metric in self.metrics]

        return dict(zip(metricNames, metricAssessments))