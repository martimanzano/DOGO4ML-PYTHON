from TrustAssessor.assessment_methods.AssessmentMethod import AssessmentMethod
from anytree import Node
from anytree.exporter import JsonExporter, DictExporter
from math import isclose

class WeightedAverage(AssessmentMethod):
    """Class that implements the trust assessment using as a weighted average. It uses the weights and hierarchy specified in the configuration file (previously loaded into the additionalProperties constructor's parameter).
    
    This class is able to process arbitrary hierarchies, although the weight's sum of each level must add to 1 to ensure the assessment's consistency.
    """
    
    def __init__(self, additionalProperties):
        """Retrieves the dict containing the hierarchical tree to use from the additionalProperties parameter and prepares the instance's attributes to perform the trust assessment.

        Args:
            additionalProperties (dict): [dictionary of parameters required by the assessment method, i.e., the hierarchical tree containing weights for each level]
        """

        super().__init__()
        self.hierarchyTree = additionalProperties
    
    def assess(self):
        """Assesses the trust as a weighted average and stores the assessment. To ensure the assessment's explainability and traceability,
        this method produces a tree containing the hierarchical assessment, with the raw weighted and unweighted metrics' assessments, as well as the upper levels of the hierarchy. It leverages the hierarchical tree computation to the "evaluate_tree" function. Stores the result as a JSON formatted string and as a dict containing the unweighted and weighted scores at each level of the tree.
        """

        trust_hiearchy_node = Node(name="Trust")

        trust_hiearchy_node.weighted_score = round(self.evaluate_tree(self.hierarchyTree, trust_hiearchy_node), 2)

        exporter = JsonExporter(indent=2)
        self.assessmentAsFormattedString = exporter.export(trust_hiearchy_node)
        self.assessment = DictExporter().export(trust_hiearchy_node)

        return trust_hiearchy_node.weighted_score

    def evaluate_tree(self, dict_node, hiearchy_parent_node):
        """Recursive function that validates and assesses a certain level of the trust hierarchy tree as a weighted average.
        When such level is not a leaf node, the assessment is performed recursively. When it is a leaf node, the assessment
        is performed by taking the raw metric's value (i.e., score) from the metrics' list and its weight extracted from the additionalProperties dict.
 
        Args:
            dict_node (dict): Current tree's level to evaluate
            hiearchy_parent_node (anytree's Node): Parent node, used to link the current evaluated node with its parent

        Raises:
            Exception: When the hierarchy's weights are not consistent, i.e., any level of the tree does not add up to 1.

        Returns:
            float: weighted score for a hierarchy level (recursively).
        """
        tree_score = node_accumulated_weight = 0
        for node in dict_node:
            node_score = node_weight = 0
            hiearchy_child_node = Node(name=node, parent=hiearchy_parent_node)
            if type(dict_node[node]) is dict:
                hiearchy_child_node.name = node.split("-")[0]
                node_weight = float(node.split("-")[1])
                hiearchy_child_node.weight = node_weight
                node_raw_score = self.evaluate_tree(dict_node[node], hiearchy_child_node)
            else:
                node_weight = dict_node[node]
                hiearchy_child_node.weight = node_weight
                node_raw_score = self.getMetricsAssessmentDict()[node]
            node_score = node_weight * node_raw_score
            tree_score += node_score
            hiearchy_child_node.weighted_score = round(node_score, ndigits=2)
            hiearchy_child_node.raw_score = round(node_raw_score, ndigits=2)
            node_accumulated_weight += node_weight
        if isclose(node_accumulated_weight, 1):
            return tree_score
        else:
            raise Exception("Validation error in configuration file: weights do not add up to 1 (" + str(node_accumulated_weight) + ") in: \n" + str(dict_node))