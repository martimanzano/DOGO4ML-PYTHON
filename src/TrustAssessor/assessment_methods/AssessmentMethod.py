class AssessmentMethod:
    """Assessment method class. Implemented assessment methods should inherit
    this class and implement their custom constructor and assess methods.
    """
    def __init__(self):
        """Initializes the relevant instance's parameters"""
        self.assessmentAsFormattedString = None
        self.assessment = None
        self.metrics = []

    def assess(self):
        """Performs the trustworthiness assessment using the assessment method (child class), stores the complet assessment and returns the corresponding to the trustworthiness itself
        """
        pass
    
    def getAssessmentAsFormattedString(self) -> str:
        """Returns the computed assessment as a string. Preferably formatted as JSON
        
        Returns:
            dict: Metrics' assessments 
        """
        return self.assessmentAsFormattedString

    def getAssessment(self):
        """Returns the computed assessment as an object to enable easy access by third-parties

        Returns:
            dict: Trust assessment as an object
        """
        return self.assessment

    def getMetricsAssessmentDict(self) -> dict:
        """Returns a dictionary of shape Metric name (str) -> Metric assessment (float)

        Returns:
            dict: Metrics' assessments
        """
        metricNames = [metric.__class__.__name__ for metric in self.metrics]
        metricAssessments = [metric.getAssessment() for metric in self.metrics]

        return dict(zip(metricNames, metricAssessments))