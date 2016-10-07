
#=====================================================
# This Module provide basic concepts and structure
# - Single Model as an process node
# - Structured Analytic Strategies as directed graph
# -
#
# The Model could be
# - Static Model (Single Frame)
# - Dynamical Model ( Time-series sequential Frames)
#

class Model(object):
    def isModel(self):
        pass

class Node(object):
    def isNode(self):
        pass


class DynamicModel(Model):
    def __init__(self):
        self.DTW
        self.RNN
        self.HMM
        pass

class StaticModel(Model):
    def __init__(self):
        pass

    def predict(self):
        pass

    def predict_proba(self):
        pass

class StrategyNetwork(object):
    def __init__(self):
        pass
    def connect(self):
        pass






def singleton(class_):
  # when the decorator be excuted
  # this variable would be encapsure in this decorater
  instances = {}
  def getinstance(*args, **kwargs):
    if class_ not in instances:
        instances[class_] = class_(*args, **kwargs)
    return instances[class_]
  return getinstance

class ProcessStrategy(Topography):
    def __init__(object, graph):
        pass



class Topography(object):
    def isTopography(self):
        pass

class ProcessUnit(Node):
    def __init__(self):
        pass


@singleton
class Memo(Node):
    '''single object recording memo
    '''
    def __init__(self, objName):
        self.boxes   = []
        self.prob    = []
        self.frameID = []
        self.objName = objName

    def add_record(self, objName,tarBox, probility, frameID):
        self.boxes.append(tarBox)
        self.prob.append(probility)
        self.frameID.append(frameID)

class MemoTopography(Topography):
    def __init__(self):
        pass

    def


class Graph(object):

    def __init__(self, graph_dict=None):
        """ initializes a graph object
            If no dictionary or None is given,
            an empty dictionary will be used
        """
        if graph_dict == None:
            graph_dict = {}
        self.__graph_dict = graph_dict

    def vertices(self):
        """ returns the vertices of a graph """
        return list(self.__graph_dict.keys())

    def edges(self):
        """ returns the edges of a graph """
        return self.__generate_edges()

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in
            self.__graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.
        """
        if vertex not in self.__graph_dict:
            self.__graph_dict[vertex] = []

    def add_edge(self, edge):
        """ assumes that edge is of type set, tuple or list;
            between two vertices can be multiple edges!
        """
        edge = set(edge)
        (vertex1, vertex2) = tuple(edge)
        if vertex1 in self.__graph_dict:
            self.__graph_dict[vertex1].append(vertex2)
        else:
            self.__graph_dict[vertex1] = [vertex2]

    def __generate_edges(self):
        """ A static method generating the edges of the
            graph "graph". Edges are represented as sets
            with one (a loop back to the vertex) or two
            vertices
        """
        edges = []
        for vertex in self.__graph_dict:
            for neighbour in self.__graph_dict[vertex]:
                if {neighbour, vertex} not in edges:
                    edges.append({vertex, neighbour})
        return edges

    def __str__(self):
        res = "vertices: "
        for k in self.__graph_dict:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return res


'''
Incepton Model on Sequence

Sequence(max_length=10) => SVM
Sequence(max_length=13) => SVM
Sequence(max_length=16) => SVM
Model_average()
'''

