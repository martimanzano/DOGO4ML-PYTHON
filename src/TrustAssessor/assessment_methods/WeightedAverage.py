from TrustAssessor.assessment_methods.AssessmentMethod import AssessmentMethod
from anytree import Node
from anytree.exporter import JsonExporter, DictExporter
from math import isclose

class WeightedAverage(AssessmentMethod):
    """Class corresponding to the Trust assessment method as a weighted average. It uses the weights and hierarchy specified
    in the YAML/JSON (previously loaded into the TrustableEntity). Therefore, it takes an instance of a TrustableEntity object,
    containing the trustworthiness metrics computed and unweighted."""
    
    def __init__(self, additionalProperties):
        """A Trust_WA initializer, corresponding to a specific TrustableEntity

        Args:
            trustable_entity (TrustableEntity): Object of type TrustableEntity for which assess the Trust as a weighted average.
            It must have its trustworthiness metrics already computed as this class is only in charge of the weighted aggregations.
        """

        super().__init__()
        self.hierarchyTree = additionalProperties
    
    def assess(self):
        """Assesses the TrustEntity as a weighted average and stores the assessment. To ensure the assessment's explainability and traceability,
        this method produces a tree containing the hierarchical assessment, with the raw scores, the weights read from the TrustEntity's configuration,
        and the weighted scores. It leverages the tree computation to the "evaluate_tree" function.
        """
        trust_hiearchy_node = Node(name="Trust")
        trust_hiearchy_node.weighted_score = trust_hiearchy_node.raw_score = round(self.evaluate_tree(self.hierarchyTree, trust_hiearchy_node), 2)
        trust_hiearchy_node.weight = 1

        exporter = JsonExporter(indent=2)
        self.assessment = exporter.export(trust_hiearchy_node)
        self.assessmentObject = DictExporter().export(trust_hiearchy_node)

    def evaluate_tree(self, dict_node, hiearchy_parent_node):
        """Recursive function that validates and assesses a certain level of the trust hierarchy as a weighted average.
        When such level is not a leaf node, the assessment is performed recursively. When it is a leaf node, the assessment
        is performed by taking the raw metric's value (i.e., score) and the specified weight.

        Args:
            dict_node (dict): Current tree's level to evaluate
            hiearchy_parent_node (Node): Parent node, used to link the current evaluated node with its parent

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