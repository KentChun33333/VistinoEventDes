

import seq2seq
from seq2seq.models import SimpleSeq2seq

model = SimpleSeq2seq(input_dim=8, hidden_dim=10, output_length=1, output_dim=8)
model.compile(loss='mse', optimizer='rmsprop')



final_model.fit([input_data_1, input_data_2], targets)  # we pass one data array per model input




class Solution(object):
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        if divisor==0:
            return MAX_INT
        positive = (dividend < 0) is (divisor < 0)
        dividend, divisor = abs(dividend), abs(divisor)
        tmp=0
        while dividend >=divisor :
            dividend-=divisor
            tmp+=1
        if not positive:
            tmp = -tmp
        return tmp