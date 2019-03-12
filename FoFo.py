class Human():
    def __init__(self, height, weight, IQ, name):

        self.height = height
        self.weight = weight 
        self.IQ     = IQ
        self.name   = name 

    def run(self, how_long):
        print( self.name, 
            'with IQ: ', 
            self.IQ,
             ' is runing for ',
              how_long, ' mins')


FOFO = Human(171, 54, 130, 'FoFo')
FOFO.run(10)
print(FOFO.weight)
