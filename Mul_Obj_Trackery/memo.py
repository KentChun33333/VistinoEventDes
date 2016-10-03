

#==============================================================================
# Author : Kent Chiu (kentchun33333@gmail.com)
#==============================================================================
#
# Notice
# - This class is used to storage the softmax model detection results
# - python -m unittest -v memo for command-line test
#
#==============================================================================

import unittest

class Memo(object):
    def __init__(self, labelnumber):
        self.boxes={}
        self.prob ={}
        for i in range(labelnumber):
            self.boxes[i]=[]
            self.prob[i]=[]

    def addNode(self, labelID, box, prob):
        assert type(box)==tuple
        self.boxes[labelID].append(box)
        self.prob[labelID].append(prob)

    def extratResult(self,labelID):
        try :
            return self.boxes[labelID], self.prob[labelID]
        except Exception as err:
            print (err)


class TestStringMethods(unittest.TestCase):

    def test_sample_1(self):
        a= Memo(4)
        a.addNode(0,(2,4,5,6),0.78)
        a.addNode(1,(12,14,15,16),0.78)
        a.addNode(2,(22,24,25,26),0.78)
        a.addNode(2,(22,24,25,26),0.78)
        a.addNode(3,(32,34,35,36),0.78)
        self.assertEqual(len(a.extratResult(1)[0]),1)
        self.assertEqual(type(a.extratResult(3)),tuple)


if __name__ == '__main__':
    pass
    #unittest.main()

    # for more detail
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStringMethods)
    unittest.TextTestRunner(verbosity=2).run(suite)

